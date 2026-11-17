"""LoRA adapters for UMA-M, attached by adapter group (IDL §4.1, Eq. 3).

config.py     LoRAConfig: groups (regexes on module names), rank, alpha, dropout
sites.py      which modules are LoRA sites, and which group each one belongs to
layers.py     LoRALinear / LoRASO3Linear / LoRAMOLE: frozen W plus s * a * B A
model.py      LoRAModel: attach, reset, activate, coefficients, save / load
uma.py        load_uma_model: fairchem checkpoint with traineval settings
interface.py  the contract the selector needs (AdaptedUMAProtocol) and a CPU stand-in
__main__.py   python -m main.LoRA --checkpoint <path>: list the sites per group
"""
from .config import LoRAConfig
from .layers import LAYER_TYPES, LoRALayer, LoRALinear, LoRAMOLE, LoRASO3Linear
from .model import LoRAModel
from .sites import Site, match_sites, site_kind
from .uma import load_uma_model

__all__ = [
    "LAYER_TYPES",
    "LoRAConfig",
    "LoRALayer",
    "LoRALinear",
    "LoRAMOLE",
    "LoRAModel",
    "LoRASO3Linear",
    "Site",
    "load_uma_model",
    "match_sites",
    "site_kind",
]
