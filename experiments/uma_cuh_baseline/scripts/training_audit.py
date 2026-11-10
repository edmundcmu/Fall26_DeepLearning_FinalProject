"""Single-GPU TorchTNT callback: measured cost, loss, finite gradient checks."""
import json
from pathlib import Path
import time
import torch
from torchtnt.framework.callback import Callback
from common import save_json


class TrainingAudit(Callback):
    def on_train_start(self, state, unit):
        self.start = time.perf_counter()
        self.folder = Path(unit.job_config.metadata.log_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "training_audit.jsonl"
        self.total = sum(p.numel() for p in unit.model.parameters())
        self.trainable = sum(p.numel() for p in unit.model.parameters() if p.requires_grad)
        torch.cuda.reset_peak_memory_stats()

    def on_train_step_end(self, state, unit):
        step = unit.train_progress.num_steps_completed
        loss = float(unit.last_loss.item())
        if not torch.isfinite(unit.last_loss):
            raise FloatingPointError("Nonfinite training loss")
        # Expensive on a 1.4B model: audit gradients only in the first two steps.
        checked = step <= 2
        gradient_tensors = 0
        if checked:
            for p in unit.model.parameters():
                if p.grad is not None:
                    gradient_tensors += 1
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError("Nonfinite gradient")
            if gradient_tensors == 0:
                raise RuntimeError("No training gradients found")
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": step, "loss": loss,
                "gradient_check": checked, "gradient_tensors_checked": gradient_tensors,
                "elapsed_seconds": time.perf_counter()-self.start,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()})+"\n")

    def on_train_end(self, state, unit):
        torch.cuda.synchronize()
        save_json(self.folder / "training_summary.json", {
            "steps_completed": unit.train_progress.num_steps_completed,
            "total_parameters": self.total, "trainable_parameters": self.trainable,
            "seconds_including_validation_and_checkpoint_callbacks": time.perf_counter()-self.start,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "device": torch.cuda.get_device_name(),
            "note": "single GPU; excludes pre-training model load; first two steps include gradient audit overhead"})
