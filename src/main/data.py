"""Pinned Cu/H frames and the locked selector split (IDL §3.2, §4.1).

PSEUDOCODE: interfaces and steps only; bodies are not implemented yet.
Reads the cell-family split prepared in i-x20/uma_cuh_baseline (500/100/200),
not the ColabFit pilot data used by uma_cuh_uma_m_baseline.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARED_DIR = REPO_ROOT / "i-x20/uma_cuh_baseline/data/prepared"


def composition_key(atoms) -> tuple[int, int]:
    """Return (n_Cu, n_H).

    Used instead of formula strings: manifest.json writes "H2Cu54" while
    ase.Atoms.get_chemical_formula() returns "Cu54H2".
    """
    # n_cu = count of atoms.numbers == 29; n_h = count of atoms.numbers == 1
    # assert n_cu + n_h == len(atoms)   (only Cu and H are expected)
    raise NotImplementedError


class CuHData:
    """Loads the prepared frames and checks them against manifest.json."""

    def __init__(self, prepared_dir: Path = PREPARED_DIR):
        # self.dir = prepared_dir
        # self.manifest = json.loads((prepared_dir / "manifest.json").read_text())
        raise NotImplementedError

    def load(self, split: str, unlock_test: bool = False) -> list:
        # assert split in {"train", "val", "test"}
        # if split == "test" and not unlock_test: refuse; only the final locked
        #     evaluation reads test (IDL §3.2)
        # path = self.dir / split / "structures.extxyz"
        # assert sha256(path) == self.manifest["file_sha256"][split]
        # frames = ase.io.read(path, ":")
        # every frame keeps atoms.info["source_index"]: the join key used by
        #     every later file (U0 predictions, probe responses, split lists)
        raise NotImplementedError


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
