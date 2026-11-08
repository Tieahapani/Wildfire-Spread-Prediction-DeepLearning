"""Shared Lightning module that trains and validates any fire model."""

from __future__ import annotations

import math
from dataclasses import asdict

import lightning
import torch
from torch import Tensor, nn

from wfspread.constants import NDWS_CHANNELS
from wfspread.data.batch import check_batch
from wfspread.eval.metrics import best_f1_threshold, evaluate
from wfspread.train.config import TrainConfig
from wfspread.train.losses import masked_bce

MONITOR = "val/all/pr_auc"


def warmup_cosine(step: int, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    """Learning-rate multiplier: linear warmup, then cosine decay to `min_ratio`.

    Args:
        step: Optimizer step, starting at 0.
        warmup_steps: Steps of linear warmup.
        total_steps: Steps in the whole run.
        min_ratio: Final multiplier, `min_lr / lr`.

    Returns:
        Multiplier for the peak learning rate.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


class FireModule(lightning.LightningModule):
    """Trains a model that maps `inputs` (B, C, H, W) to fire logits (B, H, W).

    Validation predictions for the whole split are collected and scored once per
    epoch: PR-AUC is undefined for a batch without fire, and it is not an average
    over batches.
    """

    def __init__(
        self, model: nn.Module, config: TrainConfig, num_channels: int = len(NDWS_CHANNELS)
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.num_channels = num_channels
        self._val_outputs: list[tuple[Tensor, Tensor, Tensor]] = []
        self.save_hyperparameters({**asdict(config), "num_channels": num_channels})

    def forward(self, inputs: Tensor) -> Tensor:
        """Fire logits, shape (B, H, W), float32."""
        logits = self.model(inputs)
        expected = (inputs.shape[0], *inputs.shape[-2:])
        if tuple(logits.shape) != expected:
            raise ValueError(f"model returned {tuple(logits.shape)}, expected {expected}")
        # bf16 logits leave too few distinct probabilities and create PR-AUC ties.
        return logits.float()

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Masked loss on one training batch."""
        if self.config.check_batches:
            check_batch(batch, self.num_channels)
        loss = masked_bce(self(batch["inputs"]), batch["next_fire"], self.config.pos_weight)
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=len(batch["inputs"]))
        return loss

    def on_validation_epoch_start(self) -> None:
        """Start a fresh collection of validation predictions."""
        self._val_outputs.clear()

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Log the loss and keep the predictions for scoring at epoch end."""
        if self.config.check_batches:
            check_batch(batch, self.num_channels)
        logits = self(batch["inputs"])
        loss = masked_bce(logits, batch["next_fire"], self.config.pos_weight)
        self.log("val/loss", loss, on_epoch=True, batch_size=len(batch["inputs"]))
        self._val_outputs.append(
            (torch.sigmoid(logits).cpu(), batch["next_fire"].cpu(), batch["prev_fire"].cpu())
        )

    def on_validation_epoch_end(self) -> None:
        """Score the whole validation split once."""
        if self.trainer.sanity_checking or not self._val_outputs:
            self._val_outputs.clear()
            return
        probs, next_fire, prev_fire = (
            torch.cat(parts) for parts in zip(*self._val_outputs, strict=True)
        )
        self._val_outputs.clear()
        threshold = best_f1_threshold(probs, next_fire)
        scores = evaluate(probs, next_fire, prev_fire, threshold)
        self.log_dict({f"val/{name}": value for name, value in scores.items()})
        self.log("val/threshold", threshold)

    def configure_optimizers(self) -> dict:
        """AdamW with per-step warmup and cosine decay."""
        config = self.config
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = config.warmup_epochs * max(1, total_steps // config.max_epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: warmup_cosine(step, warmup_steps, total_steps, config.min_lr / config.lr),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
