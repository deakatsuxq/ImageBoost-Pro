"""RRDBNet compatible with BasicSR/Real-ESRGAN checkpoint keys.

Architecture from Real-ESRGAN / BasicSR (BSD-3-Clause / Apache-2.0).
No torchvision or BasicSR runtime dependency is needed.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + x5 * 0.2


class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return x + self.rdb3(self.rdb2(self.rdb1(x))) * 0.2


class RRDBNet(nn.Module):
    def __init__(self, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("RRDBNet supports native scale 2 or 4")
        self.scale = scale
        self.num_block = num_block
        self.conv_first = nn.Conv2d(12 if scale == 2 else 3, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*(RRDB(num_feat, num_grow_ch) for _ in range(num_block)))
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    @property
    def halo(self) -> int:
        # Longest path: first conv + 15 convs/RRDB + body conv + four HR convs.
        # The conservative halo is aligned to pixel-unshuffle's two-pixel grid.
        return (self.num_block * 15 + 5) * (2 if self.scale == 2 else 1)

    def forward(self, x):
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)
        feature = self.conv_first(x)
        feature = feature + self.conv_body(self.body(feature))
        feature = self.lrelu(self.conv_up1(F.interpolate(feature, scale_factor=2, mode="nearest")))
        feature = self.lrelu(self.conv_up2(F.interpolate(feature, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feature)))
