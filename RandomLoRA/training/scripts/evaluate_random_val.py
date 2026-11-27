import json
from pathlib import Path

import numpy as np
import torch
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

from src.LoRA.uma import load_uma_model
from src.LoRA.config import LoRAConfig
from src.LoRA.model import LoRAModel
from src.data import CuHData, composition_key
from src.objective import EnergyCalibration

ROOT = Path("/home/ubuntu/uma_cuh_baseline")
RUN = Path("runs/random/seed41_mask41")

torch.manual_seed(41)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

frames = CuHData().load("val")
cal = EnergyCalibration.load("runs/lall/calibration.json")

print("Loading Random LoRA model...", flush=True)

model = load_uma_model(
    ROOT / "checkpoints/checkpoints/uma-m-1p1.pt",
    device="cuda",
)
lora = LoRAModel(
    model,
    LoRAConfig.from_yaml("configs/adapter_selection.yaml"),
)

lora.load_adapters(RUN / "best_adapters.pt")
model.to("cuda")
model.eval()

energies = []
forces_all = []

with torch.enable_grad():
    for i, atoms in enumerate(frames):
        clean = atoms.copy()
        clean.calc = None
        clean.info = {"charge": 0, "spin": 0}

        data = AtomicData.from_ase(
            clean,
            task_name="oc20",
            r_edges=False,
            r_energy=False,
            r_forces=False,
            r_stress=False,
            r_data_keys=["spin", "charge"],
            target_dtype=torch.float32,
        )

        batch = atomicdata_list_to_batch([data]).to("cuda")
        raw = model(batch)
        cpu_batch = batch.clone().to("cpu")

        pred = {}
        for name, task in model.tasks.items():
            if name not in raw or "oc20" not in task.datasets:
                continue
            if task.property not in ("energy", "forces"):
                continue

            value = task.normalizer.denorm(raw[name][task.property])

            if task.element_references is not None:
                zero = torch.zeros_like(value.detach().cpu())
                reference = task.element_references.undo_refs(
                    cpu_batch, zero
                )
                value = value + reference.to(
                    device=value.device,
                    dtype=value.dtype,
                )

            pred[task.property] = value.detach().cpu().numpy()

        energies.append(float(np.asarray(pred["energy"]).reshape(-1)[0]))
        forces_all.append(
            np.asarray(pred["forces"]).reshape(len(atoms), 3)
        )

        if (i + 1) % 20 == 0:
            print(f"Evaluated {i+1}/{len(frames)}", flush=True)

energies = np.asarray(energies)
forces = np.concatenate(forces_all)
source_ids = np.array([int(a.info["source_index"]) for a in frames])

dft_e = np.array([a.get_potential_energy() for a in frames])
dft_f = np.concatenate([
    a.get_forces(apply_constraint=False) for a in frames
])
species = np.concatenate([a.numbers for a in frames])

fixed_offsets = np.array([
    cal.offsets[composition_key(a)] for a in frames
])

with np.load(ROOT / "reports/U0_val/val_predictions.npz") as ref:
    assert np.array_equal(source_ids, ref["source_index"])
    u0_e = ref["energy"].copy()
    u0_f = ref["forces"].copy()

def calculate_metrics(e, f):
    return {
        "energy_raw_MAE_eV": float(np.abs(e - dft_e).mean()),
        "energy_aligned_MAE_eV": float(
            np.abs(e - fixed_offsets - dft_e).mean()
        ),
        "H_force_component_MAE_eV_per_A": float(
            np.abs(f[species == 1] - dft_f[species == 1]).mean()
        ),
        "Cu_force_component_MAE_eV_per_A": float(
            np.abs(f[species == 29] - dft_f[species == 29]).mean()
        ),
    }

result = {
    "split": "val",
    "n_frames": len(frames),
    "energy_alignment": "fixed U0 train-fitted offsets",
    "U0": calculate_metrics(u0_e, u0_f),
    "L_Random": calculate_metrics(energies, forces),
}

RUN.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    RUN / "val_predictions.npz",
    energy=energies,
    forces=forces,
    source_index=source_ids,
    offsets=np.r_[0, np.cumsum([len(a) for a in frames])],
)

(RUN / "validation_metrics.json").write_text(
    json.dumps(result, indent=2)
)

print("\n===== U0 vs Random LoRA VALIDATION =====")
print(json.dumps(result, indent=2))
print("VALIDATION EVALUATION COMPLETE", flush=True)
