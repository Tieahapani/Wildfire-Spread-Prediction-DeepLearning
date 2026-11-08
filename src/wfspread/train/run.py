"""Set up and run one training job: seeds, run folder, checkpoints and stopping."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import lightning
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.utils.data import DataLoader

from wfspread.constants import NDWS_CHANNELS
from wfspread.train.callbacks import ResourceMonitor
from wfspread.train.config import TrainConfig
from wfspread.train.module import MONITOR, FireModule

REPO_ROOT = Path(__file__).resolve().parents[3]


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    run_name: str,
    runs_dir: Path | None = None,
    accelerator: str = "auto",
    num_channels: int = len(NDWS_CHANNELS),
) -> Path:
    """Train with early stopping on validation PR-AUC and return the run folder.

    Args:
        model: Maps `inputs` (B, C, H, W) to fire logits (B, H, W).
        train_loader: Training batches in the batch format, shuffled.
        val_loader: Validation batches in the batch format, not shuffled.
        config: Training settings.
        run_name: `<model>_<tag>`, used in the run ID `YYYYMMDD-HHMM_<model>_<tag>`.
        runs_dir: Parent folder for runs; defaults to `WF_RUNS`, then `runs/`.
        accelerator: Lightning accelerator, for example "auto", "gpu" or "cpu".
        num_channels: Input channels the model expects.

    Returns:
        The run folder, holding `run.yaml`, `logs/metrics.csv` and `checkpoints/`.
    """
    lightning.seed_everything(config.seed, workers=True)
    runs_dir = runs_dir or Path(os.environ.get("WF_RUNS", REPO_ROOT / "runs"))
    run_dir = runs_dir / f"{datetime.now():%Y%m%d-%H%M}_{run_name}"
    run_dir.mkdir(parents=True)
    _write_run_info(run_dir, config, train_loader)

    trainer = lightning.Trainer(
        max_epochs=config.max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=config.precision,
        gradient_clip_val=config.grad_clip,
        num_sanity_val_steps=0,
        logger=CSVLogger(run_dir, name="logs", version=""),
        callbacks=[
            ModelCheckpoint(
                dirpath=run_dir / "checkpoints",
                filename="best-{epoch:03d}",
                monitor=MONITOR,
                mode="max",
                save_last=True,
                auto_insert_metric_name=False,
            ),
            EarlyStopping(monitor=MONITOR, mode="max", patience=config.patience),
            ResourceMonitor(),
        ],
        default_root_dir=run_dir,
    )
    trainer.fit(FireModule(model, config, num_channels), train_loader, val_loader)
    return run_dir


def _write_run_info(run_dir: Path, config: TrainConfig, train_loader: DataLoader) -> None:
    info = {
        "config": asdict(config),
        "batch_size": train_loader.batch_size,
        "git_commit": _git_commit(),
        "torch": str(torch.__version__),
        "lightning": lightning.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
    }
    (run_dir / "run.yaml").write_text(yaml.safe_dump(info, sort_keys=False))


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
