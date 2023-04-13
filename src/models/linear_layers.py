# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
import torch.nn.functional as F
from torch import Tensor, nn

import struct

class CustomLinear(nn.Module):
    def __init__(self, in_ft: int, out_ft: int, scale: int = 0):
        super().__init__()
        self.w = nn.Parameter(torch.randn((out_ft, in_ft), requires_grad=True) / out_ft ** 2)
        self.b = nn.Parameter(torch.zeros((out_ft), requires_grad=True))
        self.scale = scale
        self.qw = torch.zeros_like(self.w).to(torch.int32)
        self.qb = torch.zeros_like(self.b).to(torch.int32)

    def forward(self, x: Tensor) -> Tensor:
        if self.scale == 0:
            return F.linear(x, self.w, bias=self.b)
        return (F.linear(x, self.qw, bias=self.qb) + self.scale)//self.scale


class CustomLinearResBlock(nn.Module):
    def __init__(self, in_ft: int, out_ft: int, scale: int = 0):
        super().__init__()
        self.w = nn.Parameter(torch.zeros((out_ft, in_ft), requires_grad=True))
        self.b = nn.Parameter(torch.zeros((out_ft), requires_grad=True))
        self.scale = scale
        self.qw = torch.zeros_like(self.w).to(torch.int32)
        self.qb = torch.zeros_like(self.b).to(torch.int32)

    def forward(self, x: Tensor) -> Tensor:
        if self.scale == 0:
            return F.linear(x, self.w, bias=self.b) + x
        return (F.linear(x, self.qw, bias=self.qb) + x*self.scale + self.scale)//self.scale
