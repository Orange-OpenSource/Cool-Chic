# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Block(nn.Module):
    """ConvNeXt block"""

    def __init__(self, nf, ks=7, layer_scale=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(nf, nf, kernel_size=ks, padding=ks // 2, groups=nf)
        self.norm = LayerNorm2d(nf, eps=1e-6)
        self.pwconv1 = nn.Conv2d(nf, nf * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(nf * 4, nf, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.ones(nf, 1, 1) * layer_scale)

    def forward(self, x: Tensor):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale * x + shortcut
        return x


class ResidualBlock(nn.Module):
    def __init__(self, ni: int, nf: int, n_blocks=2, stride=1):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(ni, nf, kernel_size=3, padding=1, stride=stride),
            LayerNorm2d(nf, eps=1e-6),
            nn.GELU(),
            Block(nf),
        )
        self.identity = nn.Sequential(
            (
                nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True)
                if stride > 1
                else nn.Identity()
            ),
            nn.Conv2d(ni, nf, kernel_size=1),
        )
        # self.identity = nn.Conv2d(ni, nf, kernel_size=1, stride=stride)
        self.residual = nn.Sequential(
            *[Block(nf) for _ in range(n_blocks)],
        )

    def forward(self, x: Tensor):
        x = self.downsample(x) + self.identity(x)
        x = self.residual(x)
        return x


class Analysis(nn.Module):
    def __init__(self, in_channels=3, n_grids=7, n_feat=64, n_blocks=2):
        super().__init__()
        self.n_feat = n_feat

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(in_channels, n_feat, n_blocks),
                *[
                    ResidualBlock(n_feat, n_feat, n_blocks, stride=2)
                    for _ in range(n_grids - 1)
                ],
            ]
        )

        self.fuses = nn.ModuleList(
            [nn.Conv2d(n_feat, 1, kernel_size=1) for _ in range(n_grids)]
        )

        self.reinitialize_parameters()

    def reinitialize_parameters(self) -> None:
        for m in self.children():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> List[Tensor]:
        grids = []
        for block, fuse in zip(self.blocks, self.fuses):
            x = block(x)
            grids.append(fuse(x))
        return grids

