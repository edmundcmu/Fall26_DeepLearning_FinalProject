"""Training-only energy calibration and the shared objective (IDL Eq. 2).

PSEUDOCODE: interfaces and steps only; bodies are not implemented yet.
One definition of Eq. (2) is used everywhere: probe training, the selector's
J(S), and final adapter training. Its torch and numpy forms must agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROW_E, ROW_F_H, ROW_F_CU = 0, 1, 2


@dataclass
class EnergyCalibration:
    """Energy offsets and scales fitted on the training subset only (IDL §3.3)."""

    offsets: dict[tuple[int, int], float]  # composition_key -> mean(E_U0 - E_DFT) on train
    sigma_e: float
    sigma_h: float
    sigma_cu: float
    provenance: dict  # U0 checkpoint sha256, train file sha256, statistic used

    @classmethod
    def fit(cls, train_frames, u0_energy, u0_forces, statistic: str) -> EnergyCalibration:
        # inputs: U0 predictions on the 500 TRAIN frames only
        #     (i-x20 evaluate.py writes them to <run>/train_predictions.npz)
        # refuse frames whose source_index is not in the train manifest
        # offsets[c] = mean over train frames of composition c of (E_U0 - E_DFT)
        #     same sign as evaluate.py: aligned energy = E_U0 - offsets[c]
        # e_res    = aligned E_U0 - E_DFT for every train frame
        # f_res_h  = every H  force component of (F_U0 - F_DFT)
        # f_res_cu = every Cu force component of (F_U0 - F_DFT)
        # sigma_* = rms(res) or std(res), as locked in configs/adapter_selection.yaml
        raise NotImplementedError

    def aligned_energy(self, energy: np.ndarray, comp_keys: list[tuple[int, int]]) -> np.ndarray:
        # energy[i] - offsets[comp_keys[i]]
        raise NotImplementedError

    def save(self, path) -> None:
        # JSON; refuse to overwrite (calibration is part of the locked protocol)
        raise NotImplementedError

    @classmethod
    def load(cls, path) -> EnergyCalibration:
        raise NotImplementedError


@dataclass
class TargetStack:
    """Energies and force components of a frame list stacked into one vector.

    Per frame: [E, F_0x, F_0y, F_0z, F_1x, ...]; a Cu54H2 frame gives
    1 + 56*3 = 169 rows and a Cu54 frame 163 rows. The row metadata keeps
    DFT labels, U0 predictions and every probe response aligned.
    """

    values: np.ndarray        # (M,)
    frame: np.ndarray         # (M,) position of the row's frame in the frame list
    row_type: np.ndarray      # (M,) ROW_E / ROW_F_H / ROW_F_CU
    atom: np.ndarray          # (M,) atom index, -1 for energy rows
    source_index: np.ndarray  # (n_frames,) join key back to the dataset

    @classmethod
    def from_labels(cls, frames) -> TargetStack:
        # E from atoms.get_potential_energy(); F from atoms.get_forces(apply_constraint=False)
        # for each frame: append the energy row, then x, y, z of every atom
        #     tagged ROW_F_H or ROW_F_CU by atomic number
        raise NotImplementedError

    @classmethod
    def from_predictions(cls, frames, energy, forces, calibration: EnergyCalibration) -> TargetStack:
        # energy: (n_frames,); forces: (n_atoms_total, 3) concatenated in frame order
        #     (the layout of evaluate.py's *_predictions.npz)
        # energy rows use calibration.aligned_energy; force rows are unchanged
        raise NotImplementedError

    def assert_aligned(self, other: TargetStack) -> None:
        # identical source_index order, row_type and atom arrays
        raise NotImplementedError

    def __sub__(self, other: TargetStack) -> TargetStack:
        # assert_aligned(other); return a copy with values = self.values - other.values
        raise NotImplementedError


class SpeciesBalancedObjective:
    """L = 0.10 L_E + 0.45 L_F,H + 0.45 L_F,Cu (IDL Eq. 2).

    L_E    = mean over frames            of ((E_aligned - E_DFT) / sigma_e)^2
    L_F,H  = mean over H force components  of ((F - F_DFT) / sigma_h)^2
    L_F,Cu = mean over Cu force components of ((F - F_DFT) / sigma_cu)^2
    """

    def __init__(self, calibration: EnergyCalibration, weights=(0.10, 0.45, 0.45),
                 no_h_policy: str = "renormalize"):
        raise NotImplementedError

    def row_weights(self, stack: TargetStack) -> np.ndarray:
        """Diagonal of D in Eq. (5): sum((d * residual) ** 2) equals Eq. (2) exactly."""
        # n_e, n_h, n_cu = number of rows of each row_type in this stack
        # d[E rows]  = sqrt(w_e  / (n_e  * sigma_e ** 2))
        # d[H rows]  = sqrt(w_h  / (n_h  * sigma_h ** 2))
        # d[Cu rows] = sqrt(w_cu / (n_cu * sigma_cu ** 2))
        # if n_h == 0: apply no_h_policy (renormalize: rescale w_e, w_cu to sum to 1)
        raise NotImplementedError

    def numpy_value(self, residual: TargetStack) -> float:
        # d = row_weights(residual); return float(np.sum((d * residual.values) ** 2))
        raise NotImplementedError

    def torch_loss(self, energy, forces, batch):
        # energy: (n_sys,), forces: (n_atoms, 3) from model.predict(..., create_graph=True)
        # batch carries DFT labels, composition keys and per-atom species
        # e_res = (energy - offset[comp]) - E_DFT
        # f_res = forces - F_DFT, split into H and Cu components
        # same weights and no-H policy as row_weights(); returns a scalar tensor
        # gradients reach the adapters through F = -dE/dR (double backpropagation)
        raise NotImplementedError

    def parts(self, energy, forces, batch) -> dict[str, float]:
        # {"L_E": ..., "L_F_H": ..., "L_F_Cu": ...} for training logs
        raise NotImplementedError
