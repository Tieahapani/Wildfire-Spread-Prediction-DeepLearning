"""Callbacks that record training diagnostics, and time and memory for the efficiency results."""

from __future__ import annotations

import time
from pathlib import Path

import lightning
import torch
from torch import Tensor


class TrainingDiagnostics(lightning.Callback):
    """Logs learning rate, gradient norm and per-layer weight change every few steps.

    The gradient norm is measured before clipping, so values above `grad_clip` show
    how often clipping acts. The weight change of a layer is ||w_after - w_before|| /
    ||w_before|| over one optimizer step, for every weight tensor with two or more
    dimensions (biases and normalization scales are skipped). With AdamW, values far
    below about 1e-3 suggest a layer has stopped learning, and values far above it
    suggest steps that are too large.

    Values go straight to the logger, so they are recorded at every `every_n_steps`
    step regardless of the trainer's `log_every_n_steps`.

    Args:
        every_n_steps: Measure on optimizer steps divisible by this number.

    Raises:
        ValueError: If `every_n_steps` is below 1.
    """

    def __init__(self, every_n_steps: int = 50) -> None:
        if every_n_steps < 1:
            raise ValueError(f"every_n_steps must be at least 1, got {every_n_steps}")
        self.every_n_steps = every_n_steps
        self._step = -1
        self._metrics: dict[str, float] = {}
        self._weights_before: dict[str, Tensor] = {}

    def on_before_optimizer_step(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Record the learning rate and gradient norm, and keep the weights before the step."""
        if trainer.global_step % self.every_n_steps != 0:
            return
        grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
        grad_norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g) for g in grads])
        )
        self._step = trainer.global_step
        self._metrics = {
            "diag/lr": float(optimizer.param_groups[0]["lr"]),
            "diag/grad_norm": float(grad_norm),
        }
        self._weights_before = {
            name: weight.detach().clone() for name, weight in _layer_weights(pl_module)
        }

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Compute each layer's weight change and log everything measured for this step."""
        if not self._weights_before:
            return
        for name, weight in _layer_weights(pl_module):
            before = self._weights_before[name]
            change = torch.linalg.vector_norm(weight.detach() - before)
            self._metrics[f"diag/update/{name}"] = float(
                change / torch.linalg.vector_norm(before).clamp_min(1e-12)
            )
        if trainer.logger is not None:
            trainer.logger.log_metrics(self._metrics, step=self._step)
        self._weights_before = {}


class ResourceMonitor(lightning.Callback):
    """Logs epoch time, peak GPU memory and free system memory once per training epoch.

    On GB10 `nvidia-smi` cannot report memory, so peak memory comes from the PyTorch
    allocator, and the shared CPU-GPU pool from `MemAvailable` in /proc/meminfo.
    """

    def __init__(self) -> None:
        self._start_s = 0.0

    def on_train_epoch_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        """Start the epoch timer and reset the peak-memory counter."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._start_s = time.perf_counter()

    def on_train_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        """Log the epoch's time and memory."""
        pl_module.log("time/epoch_s", time.perf_counter() - self._start_s)
        if pl_module.device.type == "cuda":
            pl_module.log("memory/peak_gpu_gb", torch.cuda.max_memory_allocated() / 1e9)
        available_gb = _mem_available_gb()
        if available_gb is not None:
            pl_module.log("memory/available_gb", available_gb)


def _layer_weights(module: torch.nn.Module) -> list[tuple[str, Tensor]]:
    return [
        (name.removeprefix("model."), param)
        for name, param in module.named_parameters()
        if param.ndim >= 2 and param.requires_grad
    ]


def _mem_available_gb(meminfo: Path = Path("/proc/meminfo")) -> float | None:
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return None
