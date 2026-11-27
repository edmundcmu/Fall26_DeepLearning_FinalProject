"""Full six-group LoRA training for UMA-M-1.1 / Cu-H."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from fairchem.core.datasets.atomic_data import (
    AtomicData,
    atomicdata_list_to_batch,
)

from src.LoRA.uma import load_uma_model
from src.LoRA.config import LoRAConfig
from src.LoRA.model import LoRAModel
from src.data import CuHData
from src.objective import EnergyCalibration, SpeciesBalancedObjective


ROOT = Path("/home/ubuntu/uma_cuh_baseline")
DEVICE = "cuda"
RUN = Path("runs/random/seed41_mask41")
CONFIG = Path("runs/random/random_config.yaml")
CHECKPOINT = ROOT / "checkpoints/checkpoints/uma-m-1p1.pt"

SAVE_EVERY = 50
VALIDATE_EVERY = 250
PRINT_EVERY = 25


def make_graph(atoms):
    """Convert ASE input to UMA data without DFT labels."""
    clean = atoms.copy()
    clean.calc = None
    clean.info = {"charge": 0, "spin": 0}

    return AtomicData.from_ase(
        clean,
        task_name="oc20",
        r_edges=False,
        r_energy=False,
        r_forces=False,
        r_stress=False,
        r_data_keys=["spin", "charge"],
        target_dtype=torch.float32,
    )


def predict(model, graph):
    """Differentiable physical-unit energy and forces."""
    gpu_batch = atomicdata_list_to_batch([graph]).to(DEVICE)

    raw = model(gpu_batch)

    # UMA adds batch_full and atomic_numbers_full during forward.
    # Copy the completed batch to CPU for element references.
    cpu_batch = gpu_batch.clone().to("cpu")
    outputs = {}

    for name, task in model.tasks.items():
        if name not in raw or "oc20" not in task.datasets:
            continue

        prop = task.property
        if prop not in ("energy", "forces"):
            continue

        value = task.normalizer.denorm(raw[name][prop])

        # Element references are fixed constants. Add them without
        # detaching the actual model prediction.
        if task.element_references is not None:
            zero = torch.zeros_like(value.detach().cpu())
            refs = task.element_references.undo_refs(
                cpu_batch, zero
            )
            value = value + refs.to(
                device=value.device, dtype=value.dtype
            )

        outputs[prop] = value

    energy = outputs["energy"].reshape(1)
    forces = outputs["forces"].reshape(-1, 3)
    return energy, forces


def save_state(path, model, optimizer, step, best_val,
               order, cursor, rng, max_updates):
    """Atomic resumable checkpoint: adapters + optimizer, not UMA."""
    adapter_state = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    payload = {
        "step": step,
        "max_updates": max_updates,
        "best_val": best_val,
        "adapters": adapter_state,
        "optimizer": optimizer.state_dict(),
        "order": order,
        "cursor": cursor,
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }

    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def restore_state(path, model, optimizer, rng, max_updates):
    state = torch.load(
        path, map_location="cpu", weights_only=False
    )

    if state["max_updates"] != max_updates:
        raise ValueError("Training update count changed")

    named = dict(model.named_parameters())

    for name, value in state["adapters"].items():
        named[name].data.copy_(value.to(named[name].device))

    optimizer.load_state_dict(state["optimizer"])
    rng.bit_generator.state = state["numpy_rng"]
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])

    return (
        state["step"],
        state["best_val"],
        state["order"],
        state["cursor"],
    )


def validate(model, objective, frames, graphs):
    model.eval()
    total = 0.0

    # Some force heads require autograd even during evaluation.
    with torch.enable_grad():
        for frame, graph in zip(frames, graphs):
            energy, forces = predict(model, graph)
            loss = objective.torch_loss(
                energy, forces, [frame]
            )
            total += float(loss.detach())

    model.train()
    return total / len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    train_cfg = cfg["training"]
    objective_cfg = cfg["objective"]

    max_updates = int(train_cfg["max_updates"])
    seed = int(train_cfg["seed"])

    if int(train_cfg["batch_size"]) != 1:
        raise ValueError("This entrypoint requires batch_size=1")

    RUN.mkdir(parents=True, exist_ok=True)
    state_path = RUN / "latest_state.pt"

    if args.resume and not state_path.exists():
        raise FileNotFoundError(state_path)

    if not args.resume and state_path.exists():
        raise RuntimeError(
            "Existing training checkpoint. Use --resume."
        )

    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    data = CuHData()
    train_frames = data.load("train")
    val_frames = data.load("val")

    print(
        f"Data: train={len(train_frames)}, "
        f"val={len(val_frames)}",
        flush=True,
    )

    train_graphs = [make_graph(a) for a in train_frames]
    val_graphs = [make_graph(a) for a in val_frames]

    calibration = EnergyCalibration.load(
        objective_cfg["calibration_file"]
    )

    objective = SpeciesBalancedObjective(
        calibration,
        weights=(
            objective_cfg["w_energy"],
            objective_cfg["w_force_h"],
            objective_cfg["w_force_cu"],
        ),
        no_h_policy=objective_cfg["no_h_policy"],
    )

    print("Loading UMA and selected LoRA groups...", flush=True)

    model = load_uma_model(CHECKPOINT, device=DEVICE)

    lora = LoRAModel(
        model,
        LoRAConfig.from_yaml(
            cfg["adapters"]["config_file"]
        ),
    )

    groups = list(cfg["adapters"]["active_groups"])

    lora.reset(groups, seed=seed)
    lora.activate(groups)
    model.to(DEVICE)
    model.train()

    params = list(lora.adapter_parameters(groups))

    optimizer = torch.optim.AdamW(
        params,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_frames))
    cursor = 0
    start_step = 0
    best_val = float("inf")

    if args.resume:
        start_step, best_val, order, cursor = restore_state(
            state_path, model, optimizer, rng, max_updates
        )
        print(
            f"Resumed from step {start_step}",
            flush=True,
        )

    print(
        f"Trainable parameters: "
        f"{sum(p.numel() for p in params):,}",
        flush=True,
    )
    print("Training started.", flush=True)

    clock = time.perf_counter()
    log_path = RUN / "training.jsonl"

    for step in range(start_step + 1, max_updates + 1):
        if cursor >= len(order):
            order = rng.permutation(len(train_frames))
            cursor = 0

        idx = int(order[cursor])
        cursor += 1

        optimizer.zero_grad(set_to_none=True)

        energy, forces = predict(
            model, train_graphs[idx]
        )

        loss = objective.torch_loss(
            energy, forces, [train_frames[idx]]
        )

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Nonfinite loss at update {step}"
            )

        loss.backward()

        grads = [
            p.grad for p in params if p.grad is not None
        ]
        if not grads or not all(
            torch.isfinite(g).all().item() for g in grads
        ):
            raise FloatingPointError(
                f"Invalid gradients at update {step}"
            )

        optimizer.step()

        record = {
            "step": step,
            "train_loss": float(loss.detach()),
            "source_index": int(
                train_frames[idx].info["source_index"]
            ),
        }

        if step % VALIDATE_EVERY == 0:
            score = validate(
                model, objective, val_frames, val_graphs
            )
            if not np.isfinite(score):
                raise FloatingPointError(
                    "Nonfinite validation loss"
                )

            record["val_loss"] = score

            if score < best_val:
                best_val = score
                lora.save_adapters(
                    RUN / "best_adapters.pt",
                    groups,
                    metadata={
                        "step": step,
                        "val_loss": score,
                        "seed": seed,
                    },
                )

            print(
                f"Validation at {step}: "
                f"loss={score:.6f}, "
                f"best={best_val:.6f}",
                flush=True,
            )

        if step % SAVE_EVERY == 0:
            save_state(
                state_path, model, optimizer,
                step, best_val, order, cursor,
                rng, max_updates,
            )

        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if step % PRINT_EVERY == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - clock
            completed = step - start_step
            sec_per_step = elapsed / completed
            remaining = (
                max_updates - step
            ) * sec_per_step / 3600

            print(
                f"Update {step}/{max_updates} | "
                f"Loss {record['train_loss']:.6f} | "
                f"{sec_per_step:.2f} s/update | "
                f"ETA {remaining:.2f} h",
                flush=True,
            )

    lora.save_adapters(
        RUN / "final_adapters.pt",
        groups,
        metadata={
            "step": max_updates,
            "seed": seed,
        },
    )

    save_state(
        state_path, model, optimizer,
        max_updates, best_val,
        order, cursor, rng, max_updates,
    )

    print("RANDOM LORA TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
