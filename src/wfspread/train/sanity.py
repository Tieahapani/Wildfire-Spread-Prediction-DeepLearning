"""Checks that a model and the pipeline can learn and are scored correctly.

Run these on a few real batches before any hyperparameter search. Each check
trains a fresh model for a few steps with plain Adam, independent of the full
training loop, so a failure points at the model, the data or the scoring.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from wfspread.constants import IGNORE_INDEX
from wfspread.eval.metrics import pr_auc, valid_mask
from wfspread.train.losses import masked_bce

LossFn = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class SanityResult:
    """Outcome of one check.

    Attributes:
        name: Which check ran.
        value: The measured number, such as PR-AUC.
        limit: The number it was compared with.
        passed: Whether the check passed.
    """

    name: str
    value: float
    limit: float
    passed: bool


def overfit_check(
    make_model: Callable[[], nn.Module],
    tiles: dict[str, Tensor],
    steps: int = 300,
    lr: float = 1e-2,
    min_pr_auc: float = 0.9,
    device: str = "cpu",
    seed: int = 0,
) -> SanityResult:
    """Train on a handful of tiles and score on the same tiles.

    A model that cannot nearly memorize about 10 tiles has a bug in the model,
    the loss or the data.

    Args:
        make_model: Returns a fresh model mapping inputs (B, C, H, W) to logits (B, H, W).
        tiles: One batch of about 10 tiles in the batch format.
        steps: Optimizer steps.
        lr: Adam learning rate.
        min_pr_auc: PR-AUC on the same tiles needed to pass.
        device: Torch device.
        seed: Seed for the model's initial weights.

    Returns:
        PR-AUC on the training tiles.
    """
    model = _fresh(make_model, seed, device)
    _train(model, [tiles], steps, lr, device)
    value = _score(model, [tiles], device)
    return SanityResult("overfit", value, min_pr_auc, value >= min_pr_auc)


def shuffled_label_check(
    make_model: Callable[[], nn.Module],
    train_batches: Sequence[dict[str, Tensor]],
    val_batches: Sequence[dict[str, Tensor]],
    steps: int = 300,
    lr: float = 1e-2,
    max_ratio: float = 2.0,
    device: str = "cpu",
    seed: int = 0,
) -> SanityResult:
    """Train on labels shuffled across pixels and score on the real validation labels.

    Labels are shuffled pixel by pixel rather than tile by tile, because fires sit
    near the tile center and a model could otherwise learn that position honestly.
    With no real signal left, validation PR-AUC should fall to about the fire rate.
    A clearly higher score points to a bug, such as scoring against the wrong mask.

    Args:
        make_model: Returns a fresh model mapping inputs (B, C, H, W) to logits (B, H, W).
        train_batches: Training batches in the batch format.
        val_batches: Validation batches in the batch format, with real labels.
        steps: Optimizer steps.
        lr: Adam learning rate.
        max_ratio: Largest allowed PR-AUC as a multiple of the validation fire rate.
        device: Torch device.
        seed: Seed for the shuffle and the model's initial weights.

    Returns:
        Validation PR-AUC, compared with `max_ratio` times the fire rate.
    """
    generator = torch.Generator().manual_seed(seed)
    shuffled = [_shuffle_labels(batch, generator) for batch in train_batches]
    model = _fresh(make_model, seed, device)
    _train(model, shuffled, steps, lr, device)
    value = _score(model, val_batches, device)
    labels = torch.cat([b["next_fire"] for b in val_batches])
    limit = max_ratio * float((labels[valid_mask(labels)] == 1).float().mean())
    return SanityResult("shuffled_labels", value, limit, value <= limit)


def mask_check(
    model: nn.Module,
    batch: dict[str, Tensor],
    loss_fn: LossFn = masked_bce,
    device: str = "cpu",
    seed: int = 0,
) -> SanityResult:
    """Change predictions on unknown pixels and confirm the loss and PR-AUC stay the same.

    Args:
        model: Maps inputs (B, C, H, W) to logits (B, H, W).
        batch: One batch in the batch format with some unknown pixels.
        loss_fn: Loss under test.
        device: Torch device.
        seed: Seed for the replacement predictions.

    Returns:
        The largest change in loss or PR-AUC; passes if it is below 1e-6.

    Raises:
        ValueError: If the batch has no unknown pixel to test.
    """
    next_fire = batch["next_fire"].to(device)
    unknown = next_fire == IGNORE_INDEX
    if not unknown.any():
        raise ValueError("batch has no unknown pixels to test")
    with torch.no_grad():
        logits = model.to(device).eval()(batch["inputs"].to(device)).float()
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = 40 * torch.rand(logits.shape, generator=generator, device=device) - 20
    changed = torch.where(unknown, noise, logits)

    loss_change = abs(float(loss_fn(changed, next_fire) - loss_fn(logits, next_fire)))
    auc_change = abs(
        pr_auc(torch.sigmoid(changed), next_fire) - pr_auc(torch.sigmoid(logits), next_fire)
    )
    value = max(loss_change, auc_change)
    return SanityResult("mask", value, 1e-6, value < 1e-6)


def _fresh(make_model: Callable[[], nn.Module], seed: int, device: str) -> nn.Module:
    torch.manual_seed(seed)
    return make_model().to(device)


def _train(
    model: nn.Module, batches: Sequence[dict[str, Tensor]], steps: int, lr: float, device: str
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for batch in itertools.islice(itertools.cycle(batches), steps):
        logits = model(batch["inputs"].to(device))
        loss = masked_bce(logits, batch["next_fire"].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _score(model: nn.Module, batches: Sequence[dict[str, Tensor]], device: str) -> float:
    model.eval()
    probs = [torch.sigmoid(model(b["inputs"].to(device)).float()).cpu() for b in batches]
    return pr_auc(torch.cat(probs), torch.cat([b["next_fire"] for b in batches]))


def _shuffle_labels(batch: dict[str, Tensor], generator: torch.Generator) -> dict[str, Tensor]:
    next_fire = batch["next_fire"].clone()
    valid = valid_mask(next_fire)
    values = next_fire[valid]
    next_fire[valid] = values[torch.randperm(len(values), generator=generator)]
    return {**batch, "next_fire": next_fire}
