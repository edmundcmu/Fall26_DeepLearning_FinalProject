"""FA-JOINT adapter selection and its controls (IDL §4.2, Table 2). CPU only.

PSEUDOCODE: interfaces and steps only; bodies are not implemented yet.
Inputs are saved U0 predictions, DFT labels and probe responses; no model is
loaded here. Never reads checkpoint-validation or test labels.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ResponseBank:
    """Residual vectors, design matrices and weights for Eq. (5)."""

    group_ids: list[str]
    cost: np.ndarray     # (K,) trainable parameter count c_g
    r_probe: np.ndarray  # (M_p,) DFT - aligned U0 on the probe pool
    r_dev: np.ndarray    # (M_d,) same on the selector-development subset
    Q_probe: np.ndarray  # (M_p, K) design matrix, column g = response signature q_g
    Q_dev: np.ndarray    # (M_d, K)
    d_probe: np.ndarray  # (M_p,) diagonal of D on the probe pool
    d_dev: np.ndarray    # (M_d,)
    inputs_sha256: dict = field(default_factory=dict)

    @classmethod
    def load(cls, probe_index, u0_run_dir, split, calibration, objective, seed) -> ResponseBank:
        # U0 train/val predictions (evaluate.py *_predictions.npz) -> TargetStacks
        #     for probe_pool and selector_dev
        # DFT labels of the same frames -> TargetStack.from_labels
        # r = labels - aligned U0                           (residual vector, IDL §4.2)
        # for each group in probe_index (chosen seed): load q_probe, q_dev;
        #     assert rows aligned with r (source_index, row_type, atom)
        # d = objective.row_weights(stack) for each subset
        # assert all arrays finite; record sha256 of every input file
        raise NotImplementedError


class JointResidualFitter:
    """Eq. (5) on the probe pool, then J(S) on the selector-development subset."""

    def __init__(self, bank: ResponseBank, ridge_lambda: float):
        # self._cache: dict[frozenset[int], tuple[np.ndarray, float]] = {}
        raise NotImplementedError

    def fit(self, S: frozenset[int]) -> np.ndarray:
        # a*_S = argmin_{0 <= a <= 1} ||D (r - Q_S a)||^2 + lambda ||a||^2
        # A = d_probe[:, None] * Q_probe[:, sorted(S)]       (M_p, |S|)
        # b = d_probe * r_probe
        # A_aug = vstack(A, sqrt(lambda) * I_|S|);  b_aug = concat(b, zeros(|S|))
        # a = scipy.optimize.lsq_linear(A_aug, b_aug, bounds=(0, 1)).x
        # empty S -> empty a
        raise NotImplementedError

    def score(self, S: frozenset[int]) -> float:
        # J(S) = sum((d_dev * (r_dev - Q_dev[:, sorted(S)] @ fit(S))) ** 2)   # Eq. 2 on selector-dev
        # J(empty) = sum((d_dev * r_dev) ** 2): the unadapted U0 objective
        # cache by S: greedy and enumeration revisit the same sets
        raise NotImplementedError


@dataclass
class SelectionResult:
    method: str
    group_ids: list[str]
    mask: list[int]      # adapter mask over all K groups
    total_params: int
    budget: int
    log: list[dict]
    inputs_sha256: dict

    def to_json(self, path) -> None:
        # refuse to overwrite; final training reads this file
        raise NotImplementedError

    @classmethod
    def from_json(cls, path) -> SelectionResult:
        raise NotImplementedError


def feasible(S: frozenset[int], g: int, cost: np.ndarray, budget: int) -> bool:
    # g not in S and cost[list(S)].sum() + cost[g] <= budget
    raise NotImplementedError


class Selector(ABC):
    name: str

    @abstractmethod
    def select(self, bank: ResponseBank, budget: int) -> SelectionResult: ...


class GreedyForwardSelector(Selector):
    """FA-JOINT: greedy forward selection with cost-normalized marginal gain (Eq. 6)."""

    name = "FA-JOINT"

    def __init__(self, fitter: JointResidualFitter, stop_policy: str):
        raise NotImplementedError

    def select(self, bank, budget):
        # S = {}; J_cur = fitter.score(S)
        # loop:
        #     candidates = [g for g in range(K) if feasible(S, g, cost, budget)]
        #     if no candidates: break
        #     for g: J_g = fitter.score(S | {g}); gain_g = (J_cur - J_g) / cost[g]
        #         (score refits a*, so coefficients change after every addition)
        #     g* = argmax gain_g  (ties: smaller cost, then group order)
        #     if stop_policy == "positive_gain" and gain_g* <= 0: break
        #     S = S | {g*}; J_cur = J_g*
        #     log: iteration, every candidate's J and gain, chosen g*, a*_S, J_cur
        # return SelectionResult(self.name, ...)
        raise NotImplementedError


class IndependentSelector(Selector):
    """FA-Independent: singleton probe ranking, no complementarity."""

    name = "FA-Independent"

    def __init__(self, fitter: JointResidualFitter):
        raise NotImplementedError

    def select(self, bank, budget):
        # gain_g = (J(empty) - J({g})) / cost[g], each a fitted alone
        # sort groups by gain descending; add while feasible; no refit across groups
        raise NotImplementedError


class GradientRankSelector(Selector):
    """Force-Gradient: rank by scores from probes.ForceGradientScorer."""

    name = "Force-Gradient"

    def __init__(self, scores: dict[str, float]):
        raise NotImplementedError

    def select(self, bank, budget):
        # sort groups by score descending; add while feasible
        raise NotImplementedError


class RandomMaskSampler(Selector):
    """Random: predeclared feasible masks, one per paired repeat."""

    name = "Random"

    @staticmethod
    def predeclare(group_ids, cost, budget, seed, n_masks, path) -> None:
        # must run BEFORE any probe response is inspected (IDL: predeclared masks)
        # rng = numpy.random.default_rng(seed)
        # for i in range(n_masks): order = rng.permutation(K); add groups in that order while feasible
        # write masks, seed and timestamp to JSON; refuse to overwrite
        raise NotImplementedError

    def __init__(self, predeclared_path, repeat: int):
        raise NotImplementedError

    def select(self, bank, budget):
        # return the predeclared mask for this repeat; assert it is still feasible
        raise NotImplementedError


class ExactEnumerator:
    """Diagnostic only: J(S) for every feasible subset (K <= 8 -> at most 256 sets)."""

    def run(self, fitter: JointResidualFitter, bank, budget, greedy: SelectionResult) -> dict:
        # table = {S: fitter.score(S) for every S with cost[S].sum() <= budget}
        # best = argmin table
        # report J(greedy) - J(best), the greedy set's rank, and the full table
        #     (reused when comparing proxy rankings with longer-trained masks, IDL §4.3)
        raise NotImplementedError


def main():
    # args: --config, --probe-index, --u0-run, --split, --calibration, --gradient-scores, --out
    # bank = ResponseBank.load(...);  budget = config.selection.parameter_budget
    # fitter = JointResidualFitter(bank, config.selection.ridge_lambda)
    # for selector in (GreedyForwardSelector, IndependentSelector, GradientRankSelector,
    #                  RandomMaskSampler for each repeat):
    #     selector.select(bank, budget).to_json(out / f"{selector.name}.json")
    # ExactEnumerator().run(fitter, bank, budget, fa_joint_result) -> out / "enumeration.json"
    raise NotImplementedError


if __name__ == "__main__":
    main()
