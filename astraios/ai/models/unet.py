"""U-Net Architecture — shared base model for AI denoise and sharpen.

A standard encoder-decoder with skip connections. Designed for 512x512 patches
with 64px overlap for tiled inference on large astro images.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two sequential 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """U-Net with configurable depth and channel count.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for mono, 3 for color).
    out_channels : int
        Number of output channels.
    base_features : int
        Number of features in the first encoder level.
    depth : int
        Number of encoder/decoder levels.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = in_channels
        encoder_channels = []
        for i in range(depth):
            ch_out = base_features * (2**i)
            self.encoders.append(DoubleConv(ch_in, ch_out))
            self.pools.append(nn.MaxPool2d(2))
            encoder_channels.append(ch_out)
            ch_in = ch_out

        # Bottleneck
        self.bottleneck = DoubleConv(ch_in, base_features * (2**depth))

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            ch_up = base_features * (2 ** (i + 1))
            ch_skip = encoder_channels[i]
            self.upconvs.append(nn.ConvTranspose2d(ch_up, ch_skip, 2, stride=2))
            self.decoders.append(DoubleConv(ch_skip * 2, ch_skip))

        # Final 1x1 conv
        self.final = nn.Conv2d(base_features, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decode
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # Handle size mismatch from pooling
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.final(x)
