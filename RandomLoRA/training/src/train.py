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
        if config.max_updates < 1 or config.batch_size < 1:
            raise ValueError("Invalid training configuration")
        if config.lr <= 0 or config.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")

        self.model = model
        self.objective = objective
        self.config = config

    def fit(self, group_ids, train_frames,
            checkpoint_val_frames=None,
            out_dir: Path | None = None) -> TrainResult:
        import json
        import time
        import numpy as np
        import torch

        if not train_frames or out_dir is None:
            raise ValueError("Training data and output directory required")
        if self.config.relative_update_cap is not None:
            raise NotImplementedError("Probe update cap is not used in L-All")
        if self.config.checkpoint_every and checkpoint_val_frames is None:
            raise ValueError("Checkpoint selection requires validation data")
        if not callable(getattr(self.model, "predict", None)):
            raise NotImplementedError(
                "Differentiable UMA predict() has not been connected"
            )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)

        torch.manual_seed(self.config.seed)
        rng = np.random.default_rng(self.config.seed)
        order = rng.permutation(len(train_frames))
        cursor = 0

        self.model.activate(group_ids)

        params = list(self.model.adapter_parameters(group_ids))
        if not params or not all(p.requires_grad for p in params):
            raise RuntimeError("No trainable adapters")

        expected = {id(p) for p in params}
        actual = {
            id(p) for p in self.model.model.parameters()
            if p.requires_grad
        }
        if expected != actual:
            raise RuntimeError(
                "Trainable parameters are not exclusively LoRA"
            )

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.model.model.train()
        use_cuda = any(p.device.type == "cuda" for p in params)

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        result = TrainResult(
            trainable_params=sum(p.numel() for p in params)
        )

        start = time.perf_counter()
        best_val = float("inf")

        for step in range(1, self.config.max_updates + 1):
            if cursor >= len(order):
                order = rng.permutation(len(train_frames))
                cursor = 0

            indices = order[cursor:cursor+self.config.batch_size]
            cursor += len(indices)
            batch = [train_frames[int(i)] for i in indices]

            optimizer.zero_grad(set_to_none=True)

            energy, forces = self.model.predict(
                batch, create_graph=True
            )
            loss = self.objective.torch_loss(
                energy, forces, batch
            )

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Nonfinite loss at step {step}"
                )

            loss.backward()

            grads = [
                p.grad for p in params if p.grad is not None
            ]

            if not grads or not all(
                torch.isfinite(g).all().item() for g in grads
            ):
                raise FloatingPointError(
                    f"Missing/nonfinite gradients at step {step}"
                )

            grad_norm = float(torch.sqrt(sum(
                g.detach().square().sum() for g in grads
            )))

            optimizer.step()

            result.log.append({
                "step": step,
                "loss": float(loss.detach()),
                "grad_norm": grad_norm,
            })

            if (
                self.config.checkpoint_every
                and step % self.config.checkpoint_every == 0
            ):
                val_loss = self.evaluate(checkpoint_val_frames)
                result.log[-1]["validation_loss"] = val_loss

                if val_loss < best_val:
                    best_val = val_loss
                    path = out_dir / "best_adapters.pt"
                    self.model.save_adapters(
                        path, group_ids,
                        metadata={
                            "step": step,
                            "seed": self.config.seed,
                        }
                    )
                    result.best_adapter_path = path
                    result.best_step = step

                self.model.model.train()

        final_path = out_dir / "final_adapters.pt"

        self.model.save_adapters(
            final_path,
            group_ids,
            metadata={
                "step": self.config.max_updates,
                "seed": self.config.seed,
            },
        )

        result.best_adapter_path = (
            result.best_adapter_path or final_path
        )
        result.best_step = (
            result.best_step or self.config.max_updates
        )
        result.wall_seconds = time.perf_counter() - start

        if use_cuda:
            result.peak_gpu_bytes = (
                torch.cuda.max_memory_allocated()
            )

        (out_dir / "training_log.json").write_text(
            json.dumps(result.log, indent=2)
        )

        return result

    def evaluate(self, frames) -> float:
        import torch

        if not frames:
            raise ValueError("Empty validation set")

        self.model.model.eval()
        losses = []

        # Force prediction may require coordinate autograd.
        with torch.enable_grad():
            for frame in frames:
                energy, forces = self.model.predict(
                    [frame], create_graph=False
                )
                loss = self.objective.torch_loss(
                    energy, forces, [frame]
                )
                losses.append(float(loss.detach()))

        return sum(losses) / len(losses)
