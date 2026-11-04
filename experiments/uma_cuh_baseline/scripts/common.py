"""Data/metric helpers. No ML model dependency is needed for the data audit."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cell_key(a):
    return tuple(np.round(a.cell.array.reshape(-1), 6))


def fingerprint(a):
    # Atom-order-invariant wrapped coordinates in the supplied cell. This is an
    # exact duplicate check, not a claim of symmetry/near-duplicate detection.
    xyz = np.round(a.get_scaled_positions(wrap=True) % 1.0, 6)
    rows = sorted((int(z), *r) for z, r in zip(a.numbers, xyz))
    return hashlib.sha256(json.dumps([cell_key(a), rows]).encode()).hexdigest()


def force_metrics(pred, ref, numbers, offsets):
    """Componentwise MAE/RMSE, plus equally weighted per-configuration MAE."""
    result = {}
    for name, z in [("H", 1), ("Cu", 29), ("all", None)]:
        mask = np.ones(len(numbers), bool) if z is None else numbers == z
        err = pred[mask] - ref[mask]
        per_frame = []
        for start, end in zip(offsets[:-1], offsets[1:]):
            m = mask[start:end]
            if m.any():
                per_frame.append(float(np.abs(pred[start:end][m] - ref[start:end][m]).mean()))
        result[name] = {
            "n_atoms": int(mask.sum()), "n_components": int(err.size),
            "mae_eV_per_A": float(np.abs(err).mean()) if err.size else None,
            "rmse_eV_per_A": float(np.sqrt((err**2).mean())) if err.size else None,
            "configuration_mean_mae_eV_per_A": float(np.mean(per_frame)) if per_frame else None,
        }
    return result


def metrics(atoms, energies, forces, energy_offsets=None):
    reference_e = np.array([a.get_potential_energy() for a in atoms])
    reference_f = np.concatenate([a.get_forces(apply_constraint=False) for a in atoms])
    numbers = np.concatenate([a.numbers for a in atoms])
    counts = np.array([len(a) for a in atoms])
    offsets = np.r_[0, counts.cumsum()]
    error = np.asarray(energies) - reference_e
    result = {"n_configurations": len(atoms), "forces": force_metrics(
        np.concatenate(forces), reference_f, numbers, offsets),
        "raw_energy_mae_eV_per_configuration": float(np.abs(error).mean()),
        "raw_energy_rmse_eV_per_configuration": float(np.sqrt((error**2).mean())),
        "raw_energy_mae_eV_per_atom": float(np.mean(np.abs(error) / counts))}
    if energy_offsets is not None:
        corrections = np.array([energy_offsets[a.get_chemical_formula()] for a in atoms])
        aligned = error - corrections
        result["train_offset_aligned_energy_mae_eV_per_configuration"] = float(np.abs(aligned).mean())
        result["train_offset_aligned_energy_rmse_eV_per_configuration"] = float(np.sqrt((aligned**2).mean()))
    return result
