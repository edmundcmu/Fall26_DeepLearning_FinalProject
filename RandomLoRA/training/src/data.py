"""Pinned Cu/H frames and the locked selector split (IDL §3.2, §4.1).

PSEUDOCODE: interfaces and steps only; bodies are not implemented yet.
Reads the cell-family split prepared in i-x20/uma_cuh_baseline (500/100/200),
not the ColabFit pilot data used by uma_cuh_uma_m_baseline.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARED_DIR = REPO_ROOT / "i-x20/uma_cuh_baseline/data/prepared"



def composition_key(atoms) -> tuple[int, int]:
    """Return the Cu and H atom counts."""
    import numpy as np

    n_cu = int(np.count_nonzero(atoms.numbers == 29))
    n_h = int(np.count_nonzero(atoms.numbers == 1))

    if n_cu + n_h != len(atoms):
        raise ValueError("Unexpected element in Cu/H dataset")

    return n_cu, n_h


class CuHData:
    """Load the fixed DFT splits and verify data integrity."""

    def __init__(self, prepared_dir: Path = PREPARED_DIR):
        import json

        self.dir = Path(prepared_dir)
        self.manifest = json.loads(
            (self.dir / "manifest.json").read_text()
        )

    def load(self, split: str, unlock_test: bool = False) -> list:
        import hashlib
        import numpy as np
        from ase.io import read

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        if split == "test" and not unlock_test:
            raise PermissionError(
                "Test set is locked until final evaluation"
            )

        path = self.dir / split / "structures.extxyz"

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_sha = self.manifest["file_sha256"][split]

        if digest != expected_sha:
            raise ValueError(f"SHA256 mismatch for {split}")

        frames = read(str(path), index=":")

        expected_count = self.manifest["selected_counts"][split]
        if len(frames) != expected_count:
            raise ValueError(f"Incorrect frame count: {split}")

        actual_ids = [int(a.info["source_index"]) for a in frames]

        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError(f"Duplicate source_index in {split}")

        expected_ids = {
            int(item["source_index"])
            for item in self.manifest["structures"]
            if item["split"] == split
        }

        if set(actual_ids) != expected_ids:
            raise ValueError(f"Manifest mismatch for {split}")

        for atoms in frames:
            n_cu, n_h = composition_key(atoms)

            if n_cu != 54 or n_h not in (0, 2):
                raise ValueError("Unexpected composition")

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces(apply_constraint=False)

            if (
                not np.isfinite(energy)
                or forces.shape != (len(atoms), 3)
                or not np.isfinite(forces).all()
            ):
                raise ValueError("Invalid DFT labels")

        return frames


@dataclass(frozen=True)
class SelectorSplit:
    """Fixed sub-splits used only by the selector and checkpoint selection."""

    selector_dev: tuple[int, ...]    # 50 val source_index values: score J(S)
    checkpoint_val: tuple[int, ...]  # 50 val source_index values: choose final checkpoints
    probe_pool: tuple[int, ...]      # 30-50 train source_index values: probes + Eq. 5 fit
    seed: int

    @classmethod
    def create(cls, train_frames, val_frames, seed: int,
               n_dev: int = 50, n_probe: int = 40) -> SelectorSplit:
        # rng = numpy.random.default_rng(seed)
        # stratify val by composition_key so both halves keep the 64/36
        #     Cu54H2/Cu54 ratio of the validation subset (-> 32/18 each):
        #     for each composition c: shuffle its val frames;
        #         first round(n_c * n_dev / len(val)) -> selector_dev, rest -> checkpoint_val
        # probe_pool: same composition-stratified draw of n_probe frames from train
        # assert selector_dev and checkpoint_val are disjoint and sizes are exact
        raise NotImplementedError

    def save(self, path: Path) -> None:
        # refuse to overwrite: the split is locked once written
        # JSON: the three source_index lists, seed, manifest sha256, train/val file sha256
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> SelectorSplit:
        # read JSON; assert the recorded manifest sha256 still matches the pinned data
        raise NotImplementedError

    def subset(self, frames, which: str) -> list:
        # which in {"selector_dev", "checkpoint_val", "probe_pool"}
        # return frames whose info["source_index"] is in that list, in list order,
        #     so every stacked array built later has the same row order
        raise NotImplementedError
