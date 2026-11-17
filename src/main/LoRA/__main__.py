"""List the sites each adapter group captures in a UMA checkpoint.

    PYTHONPATH=src python -m main.LoRA --checkpoint <uma-m-1p1.pt>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import LoRAConfig
from .model import LoRAModel
from .uma import load_uma_model

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m main.LoRA", description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/adapter_selection.yaml")
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
