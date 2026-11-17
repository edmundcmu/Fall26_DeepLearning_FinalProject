"""Shared adapter training loop for short probes and final fresh training.

PSEUDOCODE: interfaces and steps only; bodies are not implemented yet.
Drafted by the selector lead because probes need it first; final training
(IDL §4.2-4.3) reuses it, so coordinate changes with the evaluation lead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    max_updates: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    checkpoint_every: int | None = None        # None for probes
    relative_update_cap: float | None = None   # probes only


@dataclass
class TrainResult:
    log: list[dict] = field(default_factory=list)  # step, loss parts, grad norm, cap hits, ckpt-val loss
    best_step: int | None = None
    best_adapter_path: Path | None = None
    wall_seconds: float = 0.0
    peak_gpu_bytes: int | None = None
    trainable_params: int = 0
    cap_hits: int = 0


def apply_relative_update_cap(model, group_ids, kappa: float) -> int:
    """Common relative-update cap for probes (IDL §4.1). Returns the number of clipped sites."""
    # hits = 0
    # for g in group_ids:
    #     ratios = model.relative_update(g)            # ||B A||_F / ||W||_F per site
    #     factors = {site: kappa / ratio for site, ratio in ratios.items() if ratio > kappa}
    #     model.rescale_update(g, factors); hits += len(factors)
    # return hits
    raise NotImplementedError


class AdapterTrainer:
    def __init__(self, model, objective, config: TrainConfig):
        raise NotImplementedError

    def fit(self, group_ids, train_frames, checkpoint_val_frames=None, out_dir: Path | None = None) -> TrainResult:
        # torch.manual_seed(config.seed); numpy rng(config.seed) for batch order
        # assert only adapters of group_ids require grad (IDL §4.1: backbone,
        #     conditioning, routing and heads frozen)
        # optimizer = <locked optimizer>(model.adapter_parameters(group_ids), lr, weight_decay)
        # for step in 1..max_updates:
        #     batch = next batch (no_h_policy decides how Cu54-only batches are weighted)
        #     energy, forces = model.predict(batch, create_graph=True)
        #     loss = objective.torch_loss(energy, forces, batch)
        #     optimizer.zero_grad(); loss.backward()
        #     assert loss and every adapter gradient are finite
        #     optimizer.step()
        #     if relative_update_cap: cap_hits += apply_relative_update_cap(model, group_ids, cap)
        #     log step, objective.parts(...), gradient norm
        #     if checkpoint_every and step % checkpoint_every == 0:
        #         model.save_adapters(out_dir / f"step_{step}.pt")
        #         ckpt_val = self.evaluate(checkpoint_val_frames)  # never selector-dev or test
        #         keep the step with the lowest checkpoint-validation objective
        # record wall time, peak GPU memory, trainable parameter count (cost reporting, IDL §4.3)
        raise NotImplementedError

    def evaluate(self, frames) -> float:
        # energy, forces = model.predict(frames, create_graph=False)
        #     (forces still need d/dR, but no graph for parameter gradients)
        # return objective value; no optimizer step
        raise NotImplementedError
