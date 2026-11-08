"""The batch format every loader returns and every model and training step reads.

See `docs/batch-format.md` for the meaning of each key and how inputs are prepared.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired, TypedDict

import torch
from torch import Tensor

from wfspread.constants import FIRE, IGNORE_INDEX, NDWS_CHANNELS, NO_FIRE

MASK_DTYPE = torch.int8


class Batch(TypedDict):
    """One batch of tiles.

    Attributes:
        inputs: Cleaned, normalized channels in `NDWS_CHANNELS` order,
            shape (B, C, H, W), float32.
        sample_id: Tile position in its split's original TFRecord order,
            shape (B,), int64.
        prev_fire: Day-t fire mask, shape (B, H, W), int8 in {-1, 0, 1}. Used by
            metrics and analysis only, never by models. Absent at inference.
        next_fire: Day-t+1 fire mask, shape (B, H, W), int8 in {-1, 0, 1}.
            Absent at inference.
    """

    inputs: Tensor
    sample_id: Tensor
    prev_fire: NotRequired[Tensor]
    next_fire: NotRequired[Tensor]


def check_batch(batch: Mapping[str, Tensor], num_channels: int = len(NDWS_CHANNELS)) -> None:
    """Raise if a batch breaks the batch format.

    Args:
        batch: A batch as returned by a loader.
        num_channels: Expected channel count. Lower than 13 only for runs that drop
            channels, such as the run without today's fire mask.

    Raises:
        ValueError: If a key is missing, or a tensor has the wrong shape, dtype or
            values.
    """
    for key in ("inputs", "sample_id"):
        if key not in batch:
            raise ValueError(f"batch is missing {key!r}")

    inputs = batch["inputs"]
    if inputs.dtype != torch.float32 or inputs.ndim != 4:
        raise ValueError(
            f"inputs must be float32 (B, C, H, W), got {inputs.dtype} {_shape(inputs)}"
        )
    n_tiles, n_channels, height, width = inputs.shape
    if n_channels != num_channels:
        raise ValueError(f"inputs has {n_channels} channels, expected {num_channels}")
    if not torch.isfinite(inputs).all():
        raise ValueError("inputs contains NaN or infinite values")

    sample_id = batch["sample_id"]
    if sample_id.dtype != torch.int64 or sample_id.shape != (n_tiles,):
        raise ValueError(
            f"sample_id must be int64 ({n_tiles},), got {sample_id.dtype} {_shape(sample_id)}"
        )

    for key in ("prev_fire", "next_fire"):
        if key in batch:
            _check_mask(key, batch[key], (n_tiles, height, width))


def _check_mask(key: str, mask: Tensor, shape: tuple[int, int, int]) -> None:
    if mask.dtype != MASK_DTYPE or tuple(mask.shape) != shape:
        raise ValueError(f"{key} must be int8 {shape}, got {mask.dtype} {_shape(mask)}")
    allowed = (mask == FIRE) | (mask == NO_FIRE) | (mask == IGNORE_INDEX)
    if not allowed.all():
        raise ValueError(f"{key} has values outside {{-1, 0, 1}}")


def _shape(tensor: Tensor) -> tuple[int, ...]:
    return tuple(tensor.shape)
