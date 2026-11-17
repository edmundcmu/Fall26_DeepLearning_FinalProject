"""Finding LoRA sites in a model and assigning each one to an adapter group."""
from __future__ import annotations

import re
from typing import NamedTuple

import torch
from torch import nn

from .config import LoRAConfig
from .layers import LoRALayer


class Site(NamedTuple):
    name: str  # full module name, e.g. blocks.3.edge_wise.so2_conv_1.fc_m0
    group: str
    module: nn.Module
    kind: str  # a key of layers.LAYER_TYPES


def site_kind(module: nn.Module) -> str | None:
    """Classify a module as a LoRA site; None for anything else."""
    if isinstance(module, LoRALayer):
        return None
    if isinstance(module, nn.Linear):
        return "linear"
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 3 and hasattr(module, "expand_index"):
        return "so3"  # fairchem SO3_Linear: weight (lmax + 1, out, in)
    weights = getattr(module, "weights", None)
    if isinstance(weights, torch.Tensor) and weights.ndim == 3 and hasattr(module, "global_mole_tensors"):
        return "mole"  # fairchem MOLE: weights (experts, out, in)
    return None


def match_sites(model: nn.Module, config: LoRAConfig) -> tuple[list[Site], list[str]]:
    """Return the sites each group captures and the legal sites left in no group.

    Raises if a site matches two groups (groups must be disjoint) or if the model
    already has adapters attached.
    """
    groups = {g: [re.compile(p) for p in pats] for g, pats in config.groups.items()}
    exclude = [re.compile(p) for p in config.exclude]
    sites, unassigned = [], []
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            raise ValueError("model already has LoRA adapters attached")
        kind = site_kind(module)
        if kind is None or any(p.search(name) for p in exclude):
            continue
        hits = [g for g, pats in groups.items() if any(p.search(name) for p in pats)]
        if len(hits) > 1:
            raise ValueError(f"{name} matches groups {hits}; adapter groups must be disjoint")
        if hits:
            sites.append(Site(name, hits[0], module, kind))
        else:
            unassigned.append(name)
    return sites, unassigned
