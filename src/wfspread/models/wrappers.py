"""Model wrappers for ablations that leave the batch format unchanged."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from wfspread.constants import CH


class ChannelSubset(nn.Module):
    """Feeds the wrapped model only some input channels, for ablations.

    Batches keep all channels, so the loader, training loop and metrics are
    unchanged. The wrapped model is stored as `model`, so checkpoints of a wrapped
    run have keys `model.<layer>` inside the Lightning `model.` prefix.

    Args:
        model: Model whose input channel count equals the number of kept channels.
        drop: Names from `NDWS_CHANNELS` to remove, for example ("prev_fire",).
    """

    def __init__(self, model: nn.Module, drop: Sequence[str]):
        super().__init__()
        unknown = set(drop) - set(CH)
        if unknown:
            raise ValueError(f"unknown channels {sorted(unknown)}")
        self.model = model
        keep = [index for name, index in CH.items() if name not in drop]
        self.register_buffer("keep", torch.tensor(keep), persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Logits (B, H, W) from inputs (B, C, H, W) with every channel present."""
        return self.model(inputs[:, self.keep])
