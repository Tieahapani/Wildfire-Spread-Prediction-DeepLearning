"""Losses over the pixels whose next-day fire label is known."""

from __future__ import annotations

from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits

from wfspread.constants import FIRE, IGNORE_INDEX


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
    if logits.shape != next_fire.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and next_fire {tuple(next_fire.shape)}")
    valid = next_fire != IGNORE_INDEX
    if not valid.any():
        # A mean over zero pixels is NaN, and some random crops have no valid pixel.
        # Zero keeps the step harmless while staying attached to the graph.
        return logits.float().sum() * 0.0
    target = (next_fire[valid] == FIRE).float()
    weight = logits.new_tensor(pos_weight, dtype=target.dtype)
    return binary_cross_entropy_with_logits(logits[valid].float(), target, pos_weight=weight)
