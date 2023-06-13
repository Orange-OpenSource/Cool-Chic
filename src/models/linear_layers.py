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
from utils.constants import ARMINT


class CustomLinear(nn.Module):
    def __init__(self, in_ft: int, out_ft: int, scale: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((out_ft, in_ft), requires_grad=True) / out_ft ** 2)
        self.bias = nn.Parameter(torch.zeros((out_ft), requires_grad=True))
        self.scale = scale
        self.armint = ARMINT
        if self.armint:
            self.qw = torch.zeros_like(self.weight).to(torch.int32)
            self.qb = torch.zeros_like(self.bias).to(torch.int32)
        else:
            self.qw = torch.zeros_like(self.weight).to(torch.int32).to(torch.float)
            self.qb = torch.zeros_like(self.bias).to(torch.int32).to(torch.float)

    def forward(self, x: Tensor) -> Tensor:
        if self.scale == 0:
            return F.linear(x, self.weight, bias=self.bias)
        if self.armint:
            return (F.linear(x, self.qw, bias=self.qb) + self.scale//2)//self.scale
        else:
            return torch.floor((F.linear(x, self.qw, bias=self.qb) + self.scale//2)/self.scale).to(torch.int32).to(torch.float)


class CustomLinearResBlock(nn.Module):
    def __init__(self, in_ft: int, out_ft: int, scale: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((out_ft, in_ft), requires_grad=True))
        self.bias = nn.Parameter(torch.zeros((out_ft), requires_grad=True))
        self.scale = scale
        self.armint = ARMINT
        if self.armint:
            self.qw = torch.zeros_like(self.weight).to(torch.int32)
            self.qb = torch.zeros_like(self.bias).to(torch.int32)
        else:
            self.qw = torch.zeros_like(self.weight).to(torch.int32).to(torch.float)
            self.qb = torch.zeros_like(self.bias).to(torch.int32).to(torch.float)

    def forward(self, x: Tensor) -> Tensor:
        if self.scale == 0:
            return F.linear(x, self.weight, bias=self.bias) + x
        if self.armint:
            return (F.linear(x, self.qw, bias=self.qb) + x*self.scale + self.scale//2)//self.scale
        else:
            return torch.floor((F.linear(x, self.qw, bias=self.qb) + x*self.scale + self.scale//2)/self.scale).to(torch.int32).to(torch.float)

# # class CustomAttentionBlock(nn.Module):
# #     def __init__(self, in_ft: int, out_ft: int):
# #         super().__init__()
# #         self.trunk_weight = nn.Parameter(torch.zeros((out_ft, in_ft), requires_grad=True))
# #         self.trunk_bias = nn.Parameter(torch.zeros((out_ft), requires_grad=True))

# #         self.sigmoid_weight = nn.Parameter(torch.randn((out_ft, in_ft), requires_grad=True) / out_ft ** 2)
# #         self.sigmoid_bias = nn.Parameter(torch.zeros((out_ft), requires_grad=True))


# #     def forward(self, x: Tensor) -> Tensor:
# #         trunk = F.linear(x, self.trunk_weight, bias=self.trunk_bias)
# #         sig = F.linear(x, self.sigmoid_weight, bias=self.sigmoid_bias)
# #         return trunk * torch.sigmoid(sig) + x
