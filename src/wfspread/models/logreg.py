"""Logistic regression on each pixel's 3x3 neighborhood (Huot et al. 2022).

One 3x3 convolution with a single output is exactly a logistic regression on the
pixel and its 8 neighbors (117 features for 13 channels), applied to every pixel
at once. Training it through the shared loop avoids building the feature matrix,
which would take about 26 GB for the training split.
"""

from __future__ import annotations

from torch import Tensor, nn


class LogisticRegression(nn.Module):
    """Linear model on each pixel's neighborhood; returns logits.

    Args:
        in_ch: Input channels.
        kernel_size: Neighborhood side: 3 for 3x3, as in Huot et al. 2022.
    """

    def __init__(self, in_ch: int = 13, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd; got {kernel_size}")
        self.linear = nn.Conv2d(in_ch, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, inputs: Tensor) -> Tensor:
        """Logits (B, H, W) from inputs (B, C, H, W)."""
        return self.linear(inputs).squeeze(1)
