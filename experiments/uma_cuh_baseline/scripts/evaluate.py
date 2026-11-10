"""Evaluate U0 or an adapted checkpoint on the same pinned data.

Uses full FP32 unmerged `traineval` inference, not the changing fast default.
No torch.no_grad()/inference_mode(): conservative forces require derivatives.
Energy offsets are fitted on TRAIN predictions only, separately by composition.
"""
import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from ase.io import read
from common import ROOT, sha256, save_json, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--label", required=True, help="e.g. U0 or full-transfer-seed41")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--unlock-test", action="store_true")
    p.add_argument("--selection-record", type=Path,
                   help="Frozen protocol/run decision record required before a test run")
    args = p.parse_args()
    if args.split == "test" and not (args.unlock_test and args.selection_record and args.selection_record.is_file()):
        p.error("Test is sealed: provide --unlock-test and --selection-record after fixing the run")
    if args.output.exists():
        raise FileExistsError("Do not overwrite a completed evaluation")
    import torch
    from fairchem.core import FAIRChemCalculator
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no benchmark was run")
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock = json.loads((ROOT / "source_lock.json").read_text())
    manifest_path = ROOT / "data/prepared/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    datasets = {}
    for split in ["train", args.split]:
        f = ROOT / f"data/prepared/{split}/structures.extxyz"
        assert sha256(f) == manifest["file_sha256"][split]
        datasets[split] = read(f, ":")
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha = sha256(checkpoint)
    if args.label == "U0":
        cm = json.loads((ROOT / "checkpoints/checkpoint_manifest.json").read_text())
        assert checkpoint_sha == cm["sha256"], "U0 must be the pinned untouched checkpoint"
    calc = FAIRChemCalculator.from_model_checkpoint(str(checkpoint),
                task_name=lock["task"], inference_settings="traineval", device=args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    predicted = {}
    # Write nothing to the label-bearing source objects. Strip source labels from
    # copied input objects so an evaluator cannot accidentally reuse DFT forces.
    for split, frames in datasets.items():
        energies, forces = [], []
        for i, ref in enumerate(frames):
            a = ref.copy()
            a.info = {"charge": 0, "spin": 0}
            a.calc = calc
            e = float(a.get_potential_energy())
            f = np.array(a.get_forces(apply_constraint=False), copy=True)
            if not np.isfinite(e) or not np.isfinite(f).all():
                raise FloatingPointError(f"Nonfinite prediction at {split}:{i}")
            energies.append(e); forces.append(f)
            if (i + 1) % 25 == 0:
                print(f"{split}: {i+1}/{len(frames)}", flush=True)
        predicted[split] = (np.array(energies), forces)
    offsets = defaultdict(list)
    for a, e in zip(datasets["train"], predicted["train"][0]):
        offsets[a.get_chemical_formula()].append(float(e - a.get_potential_energy()))
    offsets = {k: float(np.mean(v)) for k, v in offsets.items()}
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    e, f = predicted[args.split]
    result = metrics(datasets[args.split], e, f, offsets)
    result.update(label=args.label, split=args.split, model_checkpoint=str(checkpoint),
                  checkpoint_sha256=checkpoint_sha, task=lock["task"], inference_settings="traineval",
                  split_manifest_sha256=sha256(manifest_path), energy_offsets_fit_on_train_eV=offsets,
                  elapsed_prediction_seconds_including_train_calibration=elapsed,
                  device=args.device, force_mask="all stored atoms; mobility metadata unavailable",
                  fairchem_version=importlib.metadata.version("fairchem-core"),
                  torch_version=torch.__version__,
                  cuda_device=torch.cuda.get_device_name() if args.device == "cuda" else None,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                  selection_record_sha256=sha256(args.selection_record) if args.selection_record else None,
                  status="measured inference; not proof of diffusion/MD accuracy")
    args.output.mkdir(parents=True)
    save_json(args.output / "metrics.json", result)
    for split, (es, fs) in predicted.items():
        np.savez_compressed(args.output / f"{split}_predictions.npz", energy=es,
            forces=np.concatenate(fs), offsets=np.r_[0, np.cumsum([len(a) for a in datasets[split]])],
            source_index=np.array([a.info["source_index"] for a in datasets[split]]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
