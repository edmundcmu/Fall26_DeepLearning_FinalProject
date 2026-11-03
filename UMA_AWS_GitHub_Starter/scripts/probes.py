"""Adaptation probes and the Force-Gradient control (IDL §4.1, Table 2).

PSEUDOCODE: interfaces and steps only; bodies are not implemented yet.
GPU stage: each candidate group is trained alone for a few updates from the
original checkpoint, and the resulting change in predictions is saved as that
group's response signature q_g (IDL Eq. 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProbeRecord:
    group_id: str
    seed: int
    steps: int
    cap_hits: int
    trainable_params: int
    wall_seconds: float
    peak_gpu_bytes: int | None
    path: Path  # npz: q_probe, q_dev, row metadata, source_index order


class ProbeRunner:
    def __init__(self, model, objective, calibration, split, frames, u0_stacks, config, out_dir: Path):
        # frames: train + val frames from CuHData
        # u0_stacks: {"probe_pool": TargetStack, "selector_dev": TargetStack} of U0 predictions
        raise NotImplementedError

    def run(self, group_id: str, seed: int) -> ProbeRecord:
        # model.reset(all groups, seed); model.activate([group_id])
        # sanity: predictions on two probe frames equal U0 (zero-update equivalence)
        # result = AdapterTrainer(model, objective, TrainConfig(
        #     max_updates=probe steps (20-50), relative_update_cap=kappa, seed=seed,
        #     checkpoint_every=None)).fit([group_id], split.subset(train, "probe_pool"))
        # for name in ("probe_pool", "selector_dev"):
        #     energy, forces = model.predict(split.subset(..., name), create_graph=False)
        #     q[name] = TargetStack.from_predictions(...) - u0_stacks[name]   # IDL Eq. 4
        # save npz: q["probe_pool"], q["selector_dev"], row metadata, group_id, seed
        # model.reset([group_id], seed)  # leave the model at U0 for the next probe
        # return ProbeRecord(... cost fields from result ...)
        raise NotImplementedError

    def run_all(self, group_ids, seeds) -> list[ProbeRecord]:
        # on PSC each (group, seed) is one job-array task; this loop is the local fallback
        # write probes/index.json listing every ProbeRecord (probe cost, IDL §4.3)
        raise NotImplementedError


class ForceGradientScorer:
    """Force-Gradient control: rank groups by the force-loss gradient at initialization."""

    def __init__(self, model, objective, config):
        raise NotImplementedError

    def score(self, group_ids, frames) -> dict[str, float]:
        # frames = probe pool, so this control sees the same labels as FA-JOINT
        # for each g:
        #     model.reset([g], seed); model.activate([g])
        #         (B = 0, hence dL/dA = 0: score the B matrices)
        #     energy, forces = model.predict(frames, create_graph=True)
        #     L_F = force terms of the objective only (0.45 L_F,H + 0.45 L_F,Cu)
        #     grads = torch.autograd.grad(L_F, list(model.lora_b_parameters(g)))
        #     s_g = sum of squared Frobenius norms (definition locked in config)
        #     if force_gradient_cost_normalized: s_g /= c_g
        # save scores JSON for GradientRankSelector
        raise NotImplementedError
