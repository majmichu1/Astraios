"""RRDBNet — the Real-ESRGAN super-resolution architecture, vendored.

Why this file exists
--------------------
``super_resolution.py`` used to import ``RRDBNet`` from ``basicsr``. That
import has never once succeeded for a normal user of this application:

* ``basicsr`` is not a declared dependency, so nobody has it installed, and
* recent ``basicsr`` releases import ``torchvision.transforms.functional_tensor``,
  which torchvision removed in 0.17, so installing it does not fix things
  either.

The import failure was caught and quietly downgraded to a bicubic upsampler,
which means "AI Super-Resolution" has been plain interpolation for every
user, while the UI kept offering it as a neural upscale.

The architecture itself is small and self-contained, so vendoring it is
simpler and far more robust than depending on a package that cannot import.
The pinned official weights in ``super_resolution.MODEL_URLS`` load straight
into this module: the parameter names below are chosen to match the released
checkpoints key-for-key.

Licensing
---------
Adapted from Real-ESRGAN / BasicSR by Xintao Wang et al., BSD-3-Clause, which
is GPL-3.0 compatible. Copyright (c) 2021 Xintao Wang.
Upstream: https://github.com/xinntao/Real-ESRGAN
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualDenseBlock(nn.Module):
    """Five densely-connected convs whose output is residually scaled by 0.2.

    Each conv sees the block input concatenated with every earlier conv's
    output, which is what makes it "dense".
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # The 0.2 residual scaling is part of the trained weights' contract,
        # not a tunable: change it and the checkpoints stop reconstructing.
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: three dense blocks, residually scaled."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN generator.

    Args:
        num_in_ch: Input channels (3 for RGB).
        num_out_ch: Output channels (3 for RGB).
        scale: Net upscaling factor. 4 is the native case; 2 and 1 are reached
            by pixel-unshuffling the input first, because the network body
            always upsamples by 4.
        num_feat: Channel width of the trunk.
        num_block: Number of RRDB blocks (23 in the official checkpoints).
        num_grow_ch: Growth channels inside each dense block.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        # x2 and x1 models fold resolution into channels up front, so the
        # first conv is wider for them. This is why an x2 checkpoint will not
        # load into a model built with scale=4.
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Nearest-neighbour + conv, twice. Matches the trained weights; a
        # bilinear or transposed-conv substitute would soften the output.
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))
