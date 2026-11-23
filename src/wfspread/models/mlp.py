"""Per-pixel MLP baseline that sees each pixel's 3x3 neighborhood.

The first layer is a 3x3 convolution, so every pixel gets its own and its 8
neighbors' channels (117 features for 13 channels); the later layers are 1x1
convolutions, which are the same small MLP applied to every pixel. With
`kernel_size=1` the model sees one pixel only.
"""

from __future__ import annotations

from torch import Tensor, nn


class PixelMLP(nn.Module):
    """MLP on each pixel's neighborhood, written as convolutions so it runs on whole tiles.

    Args:
        in_ch: Input channels.
        hidden: Width of each hidden layer.
        layers: Number of hidden layers.
        kernel_size: Neighborhood side of the first layer: 3 for 3x3, 1 for one pixel.
        dropout: Dropout after each hidden layer.
    """

    def __init__(
        self,
        in_ch: int = 13,
        hidden: int = 128,
        layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if layers < 1 or kernel_size % 2 == 0:
            raise ValueError(f"need layers >= 1, odd kernel_size; got {layers}, {kernel_size}")
        blocks: list[nn.Module] = []
        width = in_ch
        for i in range(layers):
            size = kernel_size if i == 0 else 1
            blocks += [
                nn.Conv2d(width, hidden, size, padding=size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            width = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Logits (B, H, W) from inputs (B, C, H, W)."""
        return self.head(self.body(inputs)).squeeze(1)
