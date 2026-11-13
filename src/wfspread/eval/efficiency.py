"""Model size, compute, inference speed and training cost for the efficiency comparison.

Inference is timed on random inputs of the batch-format shape: speed and memory
depend on the shape, not on the values, so no data is needed.
"""

from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from wfspread.constants import NDWS_CHANNELS
from wfspread.train.module import MONITOR

TILE_SIZE = 64


@dataclass(frozen=True)
class InferenceStats:
    """Inference speed and memory at one batch size.

    Attributes:
        batch_size: Tiles per forward pass.
        ms_per_batch: Median wall time of one forward pass, milliseconds.
        ms_per_tile: `ms_per_batch / batch_size`.
        tiles_per_s: Throughput, tiles per second.
        peak_memory_gb: Peak allocated GPU memory during the passes, weights included,
            in GB; None on CPU.
    """

    batch_size: int
    ms_per_batch: float
    ms_per_tile: float
    tiles_per_s: float
    peak_memory_gb: float | None


@dataclass(frozen=True)
class TrainingCost:
    """Training time and memory read from a run's `logs/metrics.csv`.

    Attributes:
        epochs: Training epochs completed.
        train_hours: Sum of the training epochs' wall time, hours.
        best_epoch: Epoch with the highest validation PR-AUC, counting from 0.
        hours_to_best: Training time up to and including the best epoch, hours.
        peak_gpu_gb: Highest per-epoch peak GPU memory, GB; None for CPU runs.
    """

    epochs: int
    train_hours: float
    best_epoch: int
    hours_to_best: float
    peak_gpu_gb: float | None


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flops_per_tile(
    model: nn.Module, num_channels: int = len(NDWS_CHANNELS), size: int = TILE_SIZE
) -> int:
    """Floating-point operations for one forward pass on one tile.

    Counted by PyTorch's FLOP counter, which counts a multiply-add as 2 operations
    and covers matrix multiplies and convolutions, the bulk of the work in our models.

    Args:
        model: Maps inputs (B, C, H, W) to logits (B, H, W).
        num_channels: Input channels.
        size: Tile height and width, pixels.

    Returns:
        FLOPs for a single (1, C, size, size) input.
    """
    model.eval()
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(torch.zeros(1, num_channels, size, size, device=_device_of(model)))
    return counter.get_total_flops()


@torch.no_grad()
def measure_inference(
    model: nn.Module,
    batch_size: int,
    num_channels: int = len(NDWS_CHANNELS),
    size: int = TILE_SIZE,
    warmup: int = 10,
    repeats: int = 50,
) -> InferenceStats:
    """Time forward passes in eval mode on the model's device.

    The first `warmup` passes are not timed: they include one-off costs such as
    CUDA kernel selection. GPU work is synchronized before each clock reading,
    because CUDA calls return before the work finishes.

    Args:
        model: Maps inputs (B, C, H, W) to logits (B, H, W).
        batch_size: Tiles per forward pass.
        num_channels: Input channels.
        size: Tile height and width, pixels.
        warmup: Untimed passes before measuring.
        repeats: Timed passes; the median is reported.

    Returns:
        Speed and peak memory at this batch size.
    """
    device = _device_of(model)
    is_cuda = device.type == "cuda"
    model.eval()
    inputs = torch.randn(batch_size, num_channels, size, size, device=device)
    for _ in range(warmup):
        model(inputs)
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    times_ms = []
    for _ in range(repeats):
        start_s = time.perf_counter()
        model(inputs)
        if is_cuda:
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start_s) * 1e3)

    ms_per_batch = statistics.median(times_ms)
    return InferenceStats(
        batch_size=batch_size,
        ms_per_batch=ms_per_batch,
        ms_per_tile=ms_per_batch / batch_size,
        tiles_per_s=batch_size / (ms_per_batch / 1e3),
        peak_memory_gb=torch.cuda.max_memory_allocated(device) / 1e9 if is_cuda else None,
    )


def training_cost(run_dir: Path) -> TrainingCost:
    """Read training time, best epoch and peak memory from a finished run.

    Args:
        run_dir: Run folder written by `wfspread.train.run.fit`.

    Returns:
        The run's training cost.

    Raises:
        ValueError: If the run logged no epoch times or no validation PR-AUC.
    """
    rows = list(csv.DictReader((Path(run_dir) / "logs" / "metrics.csv").open()))
    epoch_s = _per_epoch(rows, "time/epoch_s")
    pr_auc = _per_epoch(rows, MONITOR)
    if not epoch_s or not pr_auc:
        raise ValueError(f"{run_dir} has no epoch times or no {MONITOR}")
    best_epoch = max(pr_auc, key=pr_auc.get)
    peak_gb = _per_epoch(rows, "memory/peak_gpu_gb")
    return TrainingCost(
        epochs=len(epoch_s),
        train_hours=sum(epoch_s.values()) / 3600,
        best_epoch=best_epoch,
        hours_to_best=sum(s for epoch, s in epoch_s.items() if epoch <= best_epoch) / 3600,
        peak_gpu_gb=max(peak_gb.values()) if peak_gb else None,
    )


def _per_epoch(rows: list[dict[str, str]], key: str) -> dict[int, float]:
    return {int(row["epoch"]): float(row[key]) for row in rows if row.get(key)}


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device
