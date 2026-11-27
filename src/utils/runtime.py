"""Seeding, device and numeric settings shared by probes and adapter training."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # Same numeric policy as evaluate.py: full FP32, no TF32 matmuls.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def select_device(prefer: str = "cuda") -> torch.device:
    # MPS is skipped on purpose: the force loss needs double backprop through
    # fairchem's UMA kernels, which this project only validates on CUDA and CPU.
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
