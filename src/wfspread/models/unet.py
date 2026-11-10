"""U-Net for next-day wildfire spread prediction.

Input:  [B, 13, 64, 64]  cleaned, normalized channels
Output: [B, 64, 64]      raw logits (apply sigmoid for probabilities)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two conv layers with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Apply two conv-bn-relu layers."""
        return self.block(x)


class Encoder(nn.Module):
    """Downsampling path: ConvBlock at each level, MaxPool between them."""

    def __init__(self, in_ch: int, channels: list[int]):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        for out_ch in channels:
            self.blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch

    def forward(self, x):
        """Return pooled output and list of skip features."""
        skips = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        return x, skips


class Decoder(nn.Module):
    """Upsampling path: upsample, concatenate skip, ConvBlock at each level."""

    def __init__(self, channels: list[int]):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.ups.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.blocks.append(ConvBlock(out_ch * 2, out_ch))

    def forward(self, x, skips):
        """Upsample and merge skip connections level by level."""
        for up, block in zip(self.ups, self.blocks, strict=False):
            x = up(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """U-Net for next-day wildfire spread prediction."""

    def __init__(
        self,
        in_ch: int = 13,
        enc_channels: list[int] | None = None,
        bottleneck_ch: int = 256,
        dropout: float = 0.1,
    ):
        if enc_channels is None:
            enc_channels = [32, 64, 128]
        super().__init__()
        self.encoder = Encoder(in_ch, enc_channels)
        self.bottleneck = ConvBlock(enc_channels[-1], bottleneck_ch)
        self.dropout = nn.Dropout2d(dropout)
        self.decoder = Decoder([bottleneck_ch, *enc_channels[::-1]])
        self.head = nn.Conv2d(enc_channels[0], 1, kernel_size=1)

    def forward(self, x):
        """Run encoder, bottleneck, decoder and head."""
        x, skips = self.encoder(x)
        x = self.dropout(self.bottleneck(x))
        x = self.decoder(x, skips)
        return self.head(x).squeeze(1)


if __name__ == "__main__":
    model = UNet()
    fake = torch.randn(2, 13, 64, 64)
    out = model(fake)
    print(f"Input:      {fake.shape}")
    print(f"Output:     {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
