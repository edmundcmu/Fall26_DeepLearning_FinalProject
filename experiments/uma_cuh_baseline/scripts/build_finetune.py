"""Generate the pinned upstream's *actual* training configuration.

Uses upstream full-backbone + NEW conservative EFS head training. It does not
implement adapters or silently claim to preserve the original output head.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import yaml
from common import ROOT, sha256, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fairchem", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "runs/ft_config")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()
    if args.steps < 2:
        p.error("Use at least two updates for the gradient smoke test")
    lock = json.loads((ROOT / "source_lock.json").read_text())
    repo, out = args.fairchem.resolve(), args.output.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    assert commit == lock["fairchem_commit"], "Use the pinned source; re-audit APIs before upgrading"
    manifest_path = ROOT / "data/prepared/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for split in ["train", "val"]:
        assert sha256(ROOT / f"data/prepared/{split}/structures.extxyz") == manifest["file_sha256"][split]
    ckpt_meta = json.loads((ROOT / "checkpoints/checkpoint_manifest.json").read_text())
    ckpt = ROOT / "checkpoints" / lock["model_path"]
    assert sha256(ckpt) == ckpt_meta["sha256"]
    subprocess.run([sys.executable, str(repo / "src/fairchem/core/scripts/create_uma_finetune_dataset.py"),
        "--train-dir", str(ROOT / "data/prepared/train"),
        "--val-dir", str(ROOT / "data/prepared/val"),
        "--output-dir", str(out), "--uma-task", lock["task"],
        "--regression-tasks", "ef", "--base-model", lock["model"],
        "--num-workers", "1"], cwd=repo, check=True)
    # Converter failures must not silently reduce the dataset.
    for failed in out.rglob("*.failed"):
        assert not failed.read_text().strip(), f"Dataset conversion failed: {failed}"
    import numpy as np
    for split in ["train", "val"]:
        count = len(np.load(out / split / "metadata.npz")["natoms"])
        assert count == manifest["selected_counts"][split], (split, count)
    cfg_path = out / "uma_sm_finetune_template.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.update(base_model_name=lock["model"], steps=args.steps, epochs=None,
               batch_size=1, lr=args.lr, max_neighbors=300,
               evaluate_every_n_steps=2 if args.steps == 2 else 100,
               checkpoint_every_n_steps=args.steps)
    cfg["job"].update(seed=args.seed, debug=True, run_name="uma_m_cuh_full_transfer",
                      run_dir=str(ROOT / "runs/training"))
    # debug=True disables external W&B logging in this upstream version.
    for key in ["train_dataloader", "eval_dataloader"]:
        cfg[key]["batch_sampler_fn"]["seed"] = args.seed
    model = cfg["runner"]["train_eval_unit"]["model"]
    # Retain base_model_name interpolation so Hydra accepts all top-level keys.
    # A local resolver prevents a second unpinned download from HF main.
    model["checkpoint_location"] = {
        "_target_": "pinned_checkpoint.resolve",
        "model_name": "${base_model_name}", "path": str(ckpt.resolve()),
        "expected_sha256": ckpt_meta["sha256"]}
    cfg["runner"]["train_eval_unit"]["clip_grad_norm"] = 10
    cfg["runner"]["callbacks"][0]["max_saved_checkpoints"] = 1
    cfg["runner"]["callbacks"].append({"_target_": "training_audit.TrainingAudit"})
    cfg["runner"]["train_eval_unit"]["print_every"] = 1 if args.steps == 2 else 10
    if args.steps == 2:
        # Upstream enforces at least one warm-up step. Do not let its default
        # five-step warm-up exceed this two-step acceptance run.
        cfg["runner"]["train_eval_unit"]["cosine_lr_scheduler_fn"]["warmup_epochs"] = 0
    data_cfg_path = out / "data/uma_conserving_data_task_energy_force.yaml"
    data_cfg = yaml.safe_load(data_cfg_path.read_text())
    # Do not calculate or fit stress: this dataset supplies no stress labels.
    data_cfg["regress_stress"] = False
    data_cfg["tasks_list"] = [t for t in data_cfg["tasks_list"] if t["name"] != "stress"]
    data_cfg_path.write_text(yaml.safe_dump(data_cfg, sort_keys=False))
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    save_json(out / "baseline_protocol.json", {
        "fairchem_commit": commit, "model": lock["model"], "task": lock["task"],
        "checkpoint_sha256": ckpt_meta["sha256"],
        "split_manifest_sha256": sha256(manifest_path),
        "training_config_sha256": sha256(cfg_path),
        "data_config_sha256": sha256(data_cfg_path),
        "steps": args.steps, "seed": args.seed, "lr": args.lr,
        "head_policy": "upstream reinitialized MLP_EFS_Head; backbone fully trainable",
        "loss": "upstream: 20 * per-atom energy MAE + 2 * mean per-atom force-vector L2 error; training-only shared RMS normalization",
        "species_balanced_training": False,
        "checkpoint_policy": "fixed final step checkpoint; validation is diagnostic; do not choose using test",
        "status": "configuration generated; not a completed training run"})
    print(f"PYTHONPATH={ROOT / 'scripts'} fairchem -c {cfg_path}")


if __name__ == "__main__":
    main()
