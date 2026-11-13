"""Losses over the pixels whose next-day fire label is known."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits

from wfspread.constants import FIRE, IGNORE_INDEX

LossFn = Callable[[Tensor, Tensor], Tensor]
LOSS_TERMS = ("bce", "focal", "dice")


def masked_bce(logits: Tensor, next_fire: Tensor, pos_weight: float = 1.0) -> Tensor:
    """Binary cross-entropy over valid pixels of the next-day fire mask.

    Args:
        logits: Fire logits, shape (B, H, W), float.
        next_fire: Day-t+1 fire mask, shape (B, H, W), values in {-1, 0, 1}.
        pos_weight: Weight on fire pixels; 1.0 gives plain BCE.

    Returns:
        Mean loss over valid pixels, a float32 scalar.

    Raises:
        ValueError: If the shapes differ.
    """
    valid = _valid_pixels(logits, next_fire)
    if not valid.any():
        return _zero(logits)
    target = (next_fire[valid] == FIRE).float()
    weight = logits.new_tensor(pos_weight, dtype=target.dtype)
    return binary_cross_entropy_with_logits(logits[valid].float(), target, pos_weight=weight)


def masked_focal(
    logits: Tensor, next_fire: Tensor, gamma: float = 2.0, alpha: float = 0.25
) -> Tensor:
    """Focal loss over valid pixels: BCE scaled down on pixels already predicted well.

    Each pixel's BCE is multiplied by `alpha_t * (1 - p_t) ** gamma`, where `p_t` is
    the predicted probability of its true label (Lin et al. 2017, eq. 5). With
    `gamma=0` and `alpha=0.5` this is half of plain BCE.

    Args:
        logits: Fire logits, shape (B, H, W), float.
        next_fire: Day-t+1 fire mask, shape (B, H, W), values in {-1, 0, 1}.
        gamma: Focusing strength; 0 turns the down-scaling off.
        alpha: Weight on fire pixels; no-fire pixels get `1 - alpha`.

    Returns:
        Mean loss over valid pixels, a float32 scalar.

    Raises:
        ValueError: If the shapes differ.
    """
    valid = _valid_pixels(logits, next_fire)
    if not valid.any():
        return _zero(logits)
    target = (next_fire[valid] == FIRE).float()
    bce = binary_cross_entropy_with_logits(logits[valid].float(), target, reduction="none")
    p_true = torch.exp(-bce)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_true) ** gamma * bce).mean()


def masked_dice(logits: Tensor, next_fire: Tensor, smooth: float = 1.0) -> Tensor:
    """Soft Dice loss over the valid pixels of the whole batch.

    Computed over the batch rather than per tile, because many tiles have no fire
    the next day and a per-tile Dice score is undefined for them.

    Args:
        logits: Fire logits, shape (B, H, W), float.
        next_fire: Day-t+1 fire mask, shape (B, H, W), values in {-1, 0, 1}.
        smooth: Added to both sides of the ratio so an empty batch stays finite.

    Returns:
        `1 - (2 * overlap + smooth) / (predicted + actual + smooth)`, a float32 scalar.

    Raises:
        ValueError: If the shapes differ.
    """
    valid = _valid_pixels(logits, next_fire)
    if not valid.any():
        return _zero(logits)
    probs = torch.sigmoid(logits[valid].float())
    target = (next_fire[valid] == FIRE).float()
    overlap = (probs * target).sum()
    return 1 - (2 * overlap + smooth) / (probs.sum() + target.sum() + smooth)


def make_loss(
    terms: Mapping[str, float],
    pos_weight: float = 1.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> LossFn:
    """Build a weighted sum of the masked losses.

    Args:
        terms: Weight per loss, keys from `LOSS_TERMS`, for example
            `{"bce": 0.4, "dice": 0.3, "focal": 0.3}`.
        pos_weight: Fire-pixel weight in the BCE term.
        focal_gamma: Focusing strength in the focal term.
        focal_alpha: Fire-pixel weight in the focal term.

    Returns:
        A function of `(logits, next_fire)` returning a float32 scalar.

    Raises:
        ValueError: If `terms` is empty, names an unknown loss or has a weight of 0 or less.
    """
    if not terms:
        raise ValueError("loss needs at least one term")
    unknown = set(terms) - set(LOSS_TERMS)
    if unknown:
        raise ValueError(f"unknown loss terms {sorted(unknown)}; choose from {LOSS_TERMS}")
    if any(weight <= 0 for weight in terms.values()):
        raise ValueError(f"loss weights must be positive, got {dict(terms)}")

    parts: dict[str, LossFn] = {
        "bce": lambda logits, y: masked_bce(logits, y, pos_weight),
        "focal": lambda logits, y: masked_focal(logits, y, focal_gamma, focal_alpha),
        "dice": masked_dice,
    }
    chosen = [(float(weight), parts[name]) for name, weight in terms.items()]

    def loss(logits: Tensor, next_fire: Tensor) -> Tensor:
        return sum(weight * part(logits, next_fire) for weight, part in chosen)

    return loss


def _valid_pixels(logits: Tensor, next_fire: Tensor) -> Tensor:
    if logits.shape != next_fire.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and next_fire {tuple(next_fire.shape)}")
    return next_fire != IGNORE_INDEX


def _zero(logits: Tensor) -> Tensor:
    # A mean over zero pixels is NaN, and some random crops have no valid pixel.
    # Zero keeps the step harmless while staying attached to the graph.
    return logits.float().sum() * 0.0
