"""LoRAModel: attaches adapters to a model by group and controls them (IDL §4.1).

Usage notes:
- Load UMA with inference_settings="traineval" (uma.load_uma_model does). The default
  "nitro" settings and the fast backends replace the SO(2) convolutions, and MoLE
  merging would drop the adapters; merging an adapted MOLE raises instead.
- fairchem's energy head keeps the force graph only in train() mode, so call
  model.train() before computing a force loss on adapter parameters.
- Dropout is applied only at sites whose inputs are rotation invariant
  (config.invariant_input); dropping single components of l > 0 features would
  break equivariance.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .config import LoRAConfig
from .layers import LAYER_TYPES, LoRALayer
from .sites import Site, match_sites

SAVE_FORMAT = 1


def _group_seed(seed: int, group: str) -> int:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


class LoRAModel:
    """Implements the adapter side of interface.AdaptedUMAProtocol (everything except predict).

    After construction every base parameter is frozen, every adapter is disabled,
    and the model output equals the unadapted model.
    """

    def __init__(self, model: nn.Module, config: LoRAConfig):
        self.model, self.config = model, config
        model.requires_grad_(False)
        sites, self.unassigned = match_sites(model, config)
        self.layers: dict[str, list[LoRALayer]] = {g: [] for g in config.groups}
        for site in sites:
            layer = self._wrap(site)
            parent, _, child = site.name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, child, layer)
            self.layers[site.group].append(layer)
        empty = [g for g, layers in self.layers.items() if not layers]
        if empty:
            raise ValueError(f"adapter groups matched no sites: {empty}")
        self.activate([])

    def _wrap(self, site: Site) -> LoRALayer:
        kwargs = {"rank": self.config.rank_for(site.group), "scaling": self.config.scaling_for(site.group),
                  "dropout": 0.0, "site": site.name, "group": site.group}
        if site.kind == "so3":
            kwargs["per_l"] = self.config.so3_per_l
        elif self.config.is_invariant_input(site.name):
            kwargs["dropout"] = self.config.dropout
        return LAYER_TYPES[site.kind](site.module, **kwargs)

    def _check(self, group_ids) -> list[str]:
        unknown = set(group_ids) - set(self.layers)
        if unknown:
            raise KeyError(f"unknown adapter groups: {sorted(unknown)}")
        return list(group_ids)

    # --- groups ------------------------------------------------------------------------

    def group_ids(self) -> list[str]:
        return list(self.layers)

    def sites(self, group_id: str) -> list[str]:
        return [layer.site for layer in self.layers[group_id]]

    def param_count(self, group_ids) -> int:
        """c_g summed over groups (IDL Eq. 3)."""
        return sum(layer.param_count for g in self._check(group_ids) for layer in self.layers[g])

    # --- state: reset, activate, coefficients ------------------------------------------

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

    # --- parameters and update size (probes, Force-Gradient control) -------------------

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

    # --- save / load -------------------------------------------------------------------

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
            "format": SAVE_FORMAT,
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

    # --- reporting ---------------------------------------------------------------------

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
