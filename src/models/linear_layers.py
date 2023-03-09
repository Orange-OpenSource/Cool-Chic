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


class CustomLinear(nn.Module):
    def __init__(self, in_ft: int, out_ft: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn((out_ft, in_ft), requires_grad=True) / out_ft ** 2)
        self.b = nn.Parameter(torch.zeros((out_ft), requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.w, bias=self.b)


class CustomLinearResBlock(nn.Module):
    def __init__(self, in_ft: int, out_ft: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros((out_ft, in_ft), requires_grad=True))
        self.b = nn.Parameter(torch.zeros((out_ft), requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.w, bias=self.b) + x
