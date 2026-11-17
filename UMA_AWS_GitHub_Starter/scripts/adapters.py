"""LoRA adapters for UMA-M, attached by adapter group (IDL §4.1, Eq. 3).

Each adapted weight W becomes W + s * a * B A with rank r, scaling s = alpha / r
(alpha defaults to r, so s = 1 as in Eq. 3) and a per-group coefficient a (1 unless
the additivity check scales it). B starts at zero, so an attached or freshly reset
adapter reproduces the unadapted model (U0) exactly.

Supported sites keep the model rotation-equivariant:
- nn.Linear (radial MLPs, scalar MLPs, unmerged SO(2) linears): the update acts on
  the same feature axis as W, so SO(2) layers keep their (+m, -m) pairing.
- SO3_Linear: one (A, B) pair per degree l, or one shared pair, applied identically
  to all 2l + 1 orders of that degree, with no bias.
- MOLE (mixture of linear experts): the update is added to the layer output and is
  shared by every expert, so it does not depend on the routing coefficients.

Usage notes:
- Load UMA with inference_settings="traineval" (as evaluate.py does). The default
  "nitro" settings and the fast backends replace the SO(2) convolutions, and MoLE
  merging would drop the adapters; merging an adapted MOLE raises instead.
- fairchem's energy head keeps the force graph only in train() mode, so call
  model.train() before computing a force loss on adapter parameters.
- Dropout is applied only at sites whose inputs are rotation invariant
  (config.invariant_input); dropping single components of l > 0 features would
  break equivariance.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Heads, conditioning, routing and the initial embedding stay frozen (IDL §4.1).
DEFAULT_EXCLUDE = (
    r"(^|\.)output_heads\.",
    r"(^|\.)energy_block\.",
    r"(^|\.)mix_csd$",
    r"(^|\.)routing_mlp\.",
    r"(^|\.)edge_degree_embedding\.",
)
# Sites whose inputs are invariant scalars, where dropout cannot break equivariance.
DEFAULT_INVARIANT_INPUT = (r"\.rad_func\.net\.\d+$", r"\.scalar_mlp\.\d+$")

SITE_KINDS = ("linear", "so3", "mole")


@dataclass
class LoRAConfig:
    groups: dict[str, list[str]]  # group id -> regexes matched against full module names
    rank: int = 2  # IDL §4.1
    alpha: float | None = None  # scaling = alpha / rank; None keeps Eq. (3): W + B A
    dropout: float = 0.0
    so3_per_l: bool = True  # one (A, B) pair per degree l instead of one shared pair
    group_ranks: dict[str, int] = field(default_factory=dict)
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    invariant_input: list[str] = field(default_factory=lambda: list(DEFAULT_INVARIANT_INPUT))

    def __post_init__(self):
        if not self.groups:
            raise ValueError("LoRAConfig needs at least one adapter group")
        self.groups = {g: [p] if isinstance(p, str) else list(p) for g, p in self.groups.items()}
        if self.rank < 1 or any(r < 1 for r in self.group_ranks.values()):
            raise ValueError("LoRA ranks must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        unknown = set(self.group_ranks) - set(self.groups)
        if unknown:
            raise ValueError(f"group_ranks names unknown groups: {sorted(unknown)}")
        for pattern in [*sum(self.groups.values(), []), *self.exclude, *self.invariant_input]:
            re.compile(pattern)

    def rank_for(self, group: str) -> int:
        return self.group_ranks.get(group, self.rank)

    def scaling_for(self, group: str) -> float:
        rank = self.rank_for(group)
        return (self.alpha if self.alpha is not None else rank) / rank

    @classmethod
    def from_dict(cls, values: dict) -> LoRAConfig:
        known = set(cls.__dataclass_fields__)
        extra = set(values) - known
        if extra:
            raise ValueError(f"unknown LoRA config keys: {sorted(extra)}")
        return cls(**{k: v for k, v in values.items() if v is not None or k == "alpha"})

    @classmethod
    def from_yaml(cls, path: str | Path, section: str = "adapters") -> LoRAConfig:
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text())[section])


def site_kind(module: nn.Module) -> str | None:
    """Classify a module as a LoRA site; None for anything else."""
    if isinstance(module, LoRALayer):
        return None
    if isinstance(module, nn.Linear):
        return "linear"
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 3 and hasattr(module, "expand_index"):
        return "so3"
    weights = getattr(module, "weights", None)
    if isinstance(weights, torch.Tensor) and weights.ndim == 3 and hasattr(module, "global_mole_tensors"):
        return "mole"
    return None


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

    def __init__(self, base: nn.Module, per_l: bool, **kwargs):
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


def _group_seed(seed: int, group: str) -> int:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


class LoRAModel:
    """Attaches LoRA adapters to a model by group and controls them.

    Implements the adapter side of adapter_interface.AdaptedUMAProtocol (everything
    except predict). After construction every base parameter is frozen, every
    adapter is disabled, and the model output equals the unadapted model.
    """

    def __init__(self, model: nn.Module, config: LoRAConfig):
        self.model, self.config = model, config
        model.requires_grad_(False)
        sites, self.unassigned = self._match_sites()
        self.layers: dict[str, list[LoRALayer]] = {g: [] for g in config.groups}
        for name, (group, module, kind) in sites.items():
            layer = self._wrap(name, group, module, kind)
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, child, layer)
            self.layers[group].append(layer)
        empty = [g for g, layers in self.layers.items() if not layers]
        if empty:
            raise ValueError(f"adapter groups matched no sites: {empty}")
        self.activate([])

    def _match_sites(self):
        groups = {g: [re.compile(p) for p in pats] for g, pats in self.config.groups.items()}
        exclude = [re.compile(p) for p in self.config.exclude]
        sites, unassigned = {}, []
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALayer):
                raise ValueError("model already has LoRA adapters attached")
            kind = site_kind(module)
            if kind is None or any(p.search(name) for p in exclude):
                continue
            hits = [g for g, pats in groups.items() if any(p.search(name) for p in pats)]
            if len(hits) > 1:
                raise ValueError(f"{name} matches groups {hits}; adapter groups must be disjoint")
            if hits:
                sites[name] = (hits[0], module, kind)
            else:
                unassigned.append(name)
        return sites, unassigned

    def _wrap(self, name: str, group: str, module: nn.Module, kind: str) -> LoRALayer:
        invariant = any(re.search(p, name) for p in self.config.invariant_input)
        kwargs = {"rank": self.config.rank_for(group), "scaling": self.config.scaling_for(group),
                  "dropout": self.config.dropout if invariant and kind != "so3" else 0.0,
                  "site": name, "group": group}
        if kind == "linear":
            return LoRALinear(module, **kwargs)
        if kind == "so3":
            return LoRASO3Linear(module, per_l=self.config.so3_per_l, **kwargs)
        return LoRAMOLE(module, **kwargs)

    def _check(self, group_ids) -> list[str]:
        unknown = set(group_ids) - set(self.layers)
        if unknown:
            raise KeyError(f"unknown adapter groups: {sorted(unknown)}")
        return list(group_ids)

    def group_ids(self) -> list[str]:
        return list(self.layers)

    def sites(self, group_id: str) -> list[str]:
        return [layer.site for layer in self.layers[group_id]]

    def param_count(self, group_ids) -> int:
        """c_g summed over groups (IDL Eq. 3)."""
        return sum(layer.param_count for g in self._check(group_ids) for layer in self.layers[g])

    def reset(self, group_ids, seed: int) -> None:
        """Fresh initialization: A random (seeded per group), B = 0, coefficient 1."""
        for g in self._check(group_ids):
            generator = torch.Generator().manual_seed(_group_seed(seed, g))
            for layer in self.layers[g]:
                layer.reset(generator)

    def activate(self, group_ids) -> None:
        """Only these groups contribute to the output and receive gradients."""
        active = set(self._check(group_ids))
        for g, layers in self.layers.items():
            for layer in layers:
                layer.enabled = g in active
                layer.lora_A.requires_grad_(g in active)
                layer.lora_B.requires_grad_(g in active)

    def active_groups(self) -> list[str]:
        return [g for g, layers in self.layers.items() if layers[0].enabled]

    def set_coefficients(self, coeffs: dict[str, float]) -> None:
        """Scale each group's Delta W by a_g (additivity check, IDL §4.3)."""
        for g, a in coeffs.items():
            for layer in self.layers[self._check([g])[0]]:
                layer.coefficient = float(a)

    def adapter_parameters(self, group_ids) -> Iterator[nn.Parameter]:
        for g in self._check(group_ids):
            for layer in self.layers[g]:
                yield layer.lora_A
                yield layer.lora_B

    def lora_b_parameters(self, group_id: str) -> Iterator[nn.Parameter]:
        for layer in self.layers[self._check([group_id])[0]]:
            yield layer.lora_B

    def relative_update(self, group_id: str) -> dict[str, float]:
        return {layer.site: layer.relative_update() for layer in self.layers[group_id]}

    def rescale_update(self, group_id: str, factors: dict[str, float]) -> None:
        by_site = {layer.site: layer for layer in self.layers[group_id]}
        for site, factor in factors.items():
            by_site[site].rescale(factor)

    def adapter_state_dict(self, group_ids=None) -> dict[str, torch.Tensor]:
        state = {}
        for g in self._check(group_ids if group_ids is not None else self.layers):
            for layer in self.layers[g]:
                state[f"{layer.site}.lora_A"] = layer.lora_A.detach().cpu().clone()
                state[f"{layer.site}.lora_B"] = layer.lora_B.detach().cpu().clone()
        return state

    def save_adapters(self, path: str | Path, group_ids=None, metadata: dict | None = None) -> None:
        groups = self._check(group_ids if group_ids is not None else self.layers)
        torch.save({
            "format": 1,
            "config": asdict(self.config),
            "sites": {g: self.sites(g) for g in groups},
            "coefficients": {g: self.layers[g][0].coefficient for g in groups},
            "active": [g for g in self.active_groups() if g in groups],
            "state": self.adapter_state_dict(groups),
            "metadata": metadata or {},
        }, path)

    def load_adapters(self, path: str | Path) -> dict:
        """Load saved adapters into the same sites, then activate the saved groups."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for g, saved_sites in payload["sites"].items():
            if self.sites(self._check([g])[0]) != saved_sites:
                raise ValueError(f"group {g}: saved sites differ from this model's sites")
        with torch.no_grad():
            for g in payload["sites"]:
                for layer in self.layers[g]:
                    layer.lora_A.copy_(payload["state"][f"{layer.site}.lora_A"])
                    layer.lora_B.copy_(payload["state"][f"{layer.site}.lora_B"])
        self.set_coefficients(payload["coefficients"])
        self.activate(payload["active"])
        return payload["metadata"]

    def parameter_summary(self) -> dict:
        params = list(self.model.parameters())
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "trainable_fraction": trainable / total if total else 0.0,
                "per_group": {g: self.param_count([g]) for g in self.layers}}

    def describe(self) -> list[dict]:
        return [{"group": g, "site": layer.site, "kind": layer.kind, "in": layer.in_features,
                 "out": layer.out_features, "slices": layer.n_slices, "rank": layer.rank,
                 "params": layer.param_count}
                for g, layers in self.layers.items() for layer in layers]


def load_uma_model(checkpoint: str | Path, device: str = "cpu") -> nn.Module:
    """The raw UMA nn.Module (backbone + heads) with the unmerged traineval settings."""
    from fairchem.core.units.mlip_unit import load_predict_unit

    unit = load_predict_unit(checkpoint, inference_settings="traineval", device=device)
    return unit.model.module


def main() -> None:
    # Checkpoint inspection: list the sites each adapter group captures.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/adapter_selection.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    lora = LoRAModel(load_uma_model(args.checkpoint, args.device), LoRAConfig.from_yaml(args.config))
    for row in lora.describe():
        print(f"{row['group']:<10} {row['kind']:<6} {row['site']:<62} "
              f"{row['in']:>5}->{row['out']:<5} x{row['slices']} r={row['rank']} {row['params']:>9,}")
    summary = lora.parameter_summary()
    print("\nparameters per group:", {g: f"{n:,}" for g, n in summary["per_group"].items()})
    print(f"model parameters: {summary['total']:,}")
    if lora.unassigned:
        print(f"\n{len(lora.unassigned)} legal sites are in no group:")
        for name in lora.unassigned:
            print("  ", name)


if __name__ == "__main__":
    main()
