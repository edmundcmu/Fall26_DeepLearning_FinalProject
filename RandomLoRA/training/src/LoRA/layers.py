"""LoRA wrappers for the three kinds of UMA weight (IDL Eq. 3).

Each adapted weight W becomes W + s * a * B A with rank r, scaling s = alpha / r
(alpha defaults to r, so s = 1 as in Eq. 3) and a per-group coefficient a (1 unless
the additivity check scales it). B starts at zero, so a freshly reset adapter
reproduces the unadapted model (U0) exactly.

Every wrapper keeps the model rotation-equivariant:
- LoRALinear (radial MLPs, scalar MLPs, unmerged SO(2) linears): the update acts on
  the same feature axis as W, so SO(2) layers keep their (+m, -m) pairing.
- LoRASO3Linear: one (A, B) pair per degree l, or one shared pair, applied identically
  to all 2l + 1 orders of that degree, with no bias.
- LoRAMOLE (mixture of linear experts): the update is added to the layer output and is
  shared by every expert, so it does not depend on the routing coefficients.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALayer(nn.Module):
    """A frozen base layer plus the low-rank update s * a * B A (B: out x r, A: r x in)."""

    kind = ""

    def __init__(self, base: nn.Module, reference: torch.Tensor, *, in_features: int, out_features: int,
                 n_slices: int, rank: int, scaling: float, dropout: float, site: str, group: str):
        super().__init__()
        self.base_layer = base.requires_grad_(False)
        self.in_features, self.out_features = in_features, out_features
        self.n_slices, self.rank, self.scaling = n_slices, rank, scaling
        self.site, self.group = site, group
        factory = {"dtype": reference.dtype, "device": reference.device}
        self.lora_A = nn.Parameter(torch.zeros(n_slices, rank, in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(n_slices, out_features, rank, **factory))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = False
        self.coefficient = 1.0

    def __getattr__(self, name: str):
        # fairchem reads attributes such as .bias or .global_mole_tensors off these layers.
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self.__dict__.get("_modules", {}).get("base_layer")
            if base is None:
                raise
            return getattr(base, name)

    @property
    def param_count(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    def delta_weight(self) -> torch.Tensor:
        """The (n_slices, out, in) update s * a * B A."""
        return (self.scaling * self.coefficient) * (self.lora_B @ self.lora_A)

    def _low_rank(self, x: torch.Tensor) -> torch.Tensor:
        h = F.linear(self.lora_dropout(x), self.lora_A[0])
        return (self.scaling * self.coefficient) * F.linear(h, self.lora_B[0])

    @torch.no_grad()
    def reset(self, generator: torch.Generator) -> None:
        # Kaiming-uniform A (bound 1/sqrt(in), as in LoRA/PEFT) and B = 0.
        bound = 1.0 / math.sqrt(self.in_features)
        noise = torch.rand(self.lora_A.shape, generator=generator, dtype=torch.float64)
        self.lora_A.copy_(((2 * noise - 1) * bound).to(self.lora_A))
        self.lora_B.zero_()
        self.coefficient = 1.0

    def reference_norm(self) -> torch.Tensor:
        raise NotImplementedError

    def update_norm(self) -> torch.Tensor:
        return self.delta_weight().norm()

    @torch.no_grad()
    def relative_update(self) -> float:
        """||Delta W||_F / ||W||_F, the quantity the probe update cap bounds."""
        return float(self.update_norm() / self.reference_norm())

    @torch.no_grad()
    def rescale(self, factor: float) -> None:
        self.lora_B.mul_(factor)  # Delta W = B A scales linearly with B

    def extra_repr(self) -> str:
        return f"group={self.group}, rank={self.rank}, slices={self.n_slices}, enabled={self.enabled}"


class LoRALinear(LoRALayer):
    kind = "linear"

    def __init__(self, base: nn.Linear, **kwargs):
        super().__init__(base, base.weight, in_features=base.in_features,
                         out_features=base.out_features, n_slices=1, **kwargs)

    @property
    def weight(self) -> torch.Tensor:
        # Some SO(2) code paths read .weight instead of calling forward.
        w = self.base_layer.weight
        return w + self.delta_weight()[0] if self.enabled else w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        return out + self._low_rank(x) if self.enabled else out

    def reference_norm(self) -> torch.Tensor:
        return self.base_layer.weight.norm()


class LoRASO3Linear(LoRALayer):
    kind = "so3"

    def __init__(self, base: nn.Module, per_l: bool = True, **kwargs):
        n_l, out_features, in_features = base.weight.shape
        super().__init__(base, base.weight, in_features=in_features, out_features=out_features,
                         n_slices=n_l if per_l else 1, **kwargs)
        self.n_degrees = n_l

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, (lmax + 1)^2, C_in); every order of degree l shares that degree's update.
        out = self.base_layer(x)
        if not self.enabled:
            return out
        delta = self.delta_weight()
        if self.n_slices == 1:
            return out + torch.einsum("bmi, oi -> bmo", x, delta[0])
        delta = delta.index_select(0, self.base_layer.expand_index)
        return out + torch.einsum("bmi, moi -> bmo", x, delta)

    def update_norm(self) -> torch.Tensor:
        # A shared update is applied to all lmax + 1 degree slices of W.
        scale = 1.0 if self.n_slices > 1 else math.sqrt(self.n_degrees)
        return self.delta_weight().norm() * scale

    def reference_norm(self) -> torch.Tensor:
        return self.base_layer.weight.norm()


class LoRAMOLE(LoRALayer):
    kind = "mole"

    def __init__(self, base: nn.Module, **kwargs):
        _, out_features, in_features = base.weights.shape
        super().__init__(base, base.weights, in_features=in_features,
                         out_features=out_features, n_slices=1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        return out + self._low_rank(x) if self.enabled else out

    def merged_linear_layer(self):
        raise RuntimeError(f"{self.site}: merging MoLE experts would drop the LoRA update; "
                           "use unmerged inference (inference_settings='traineval')")

    def reference_norm(self) -> torch.Tensor:
        # The merged weight depends on the system's routing; use the RMS expert norm.
        return self.base_layer.weights.flatten(1).norm(dim=1).pow(2).mean().sqrt()


# Site kind (see sites.site_kind) -> the wrapper that adapts it.
LAYER_TYPES: dict[str, type[LoRALayer]] = {cls.kind: cls for cls in (LoRALinear, LoRASO3Linear, LoRAMOLE)}
