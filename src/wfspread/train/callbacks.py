"""Callbacks that record time and memory for the efficiency results."""

from __future__ import annotations

import time
from pathlib import Path

import lightning
import torch


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


def _mem_available_gb(meminfo: Path = Path("/proc/meminfo")) -> float | None:
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return None
