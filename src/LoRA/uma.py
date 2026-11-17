"""Loading UMA from fairchem in the form the adapters need."""
from __future__ import annotations

from pathlib import Path

from torch import nn


def load_uma_model(checkpoint: str | Path, device: str = "cpu") -> nn.Module:
    """The raw UMA nn.Module (backbone + heads) with the unmerged traineval settings.

    The default "nitro" settings would replace the SO(2) convolutions with fused
    kernels and merge the MoLE experts, which bypasses the adapters silently.
    """
    from fairchem.core.units.mlip_unit import load_predict_unit

    unit = load_predict_unit(checkpoint, inference_settings="traineval", device=device)
    return unit.model.module
