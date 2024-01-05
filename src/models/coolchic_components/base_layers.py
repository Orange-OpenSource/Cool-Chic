# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
import torch.nn.functional as F
from torch import Tensor, nn
from utils.misc import ARMINT


# ===================== Linear (MLP) layers for the ARM ===================== #
class CustomLinear(nn.Module):
    def __init__(self, in_ft: int, out_ft: int, scale: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ft, in_ft, requires_grad=True) / out_ft ** 2)
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
# ===================== Linear (MLP) layers for the ARM ===================== #


# =================== Conv layer layers for the Synthesis =================== #
class SynthesisLayer(nn.Module):
    def __init__(
        self,
        input_ft: int,
        output_ft: int,
        kernel_size: int,
        non_linearity: nn.Module = nn.Identity()
    ):
        """Instantiate a synthesis layer.

        Args:
            input_ft (int): Input feature
            output_ft (int): Output feature
            kernel_size (int): Kernel size
            non_linearity (nn.Module): Non linear function applied at the very end
                of the forward. Defaults to nn.Identity()
        """
        super().__init__()

        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        self.conv_layer = nn.Conv2d(
            input_ft,
            output_ft,
            kernel_size
        )

        self.non_linearity = non_linearity

        # More stable if initialized as a zero-bias layer with smaller variance
        # for the weights.
        with torch.no_grad():
            self.conv_layer.weight.data = self.conv_layer.weight.data / output_ft ** 2
            self.conv_layer.bias.data = self.conv_layer.bias.data * 0.

    def forward(self, x: Tensor) -> Tensor:
        return self.non_linearity(self.conv_layer(self.pad(x)))

class SynthesisResidualLayer(nn.Module):
    def __init__(
        self,
        input_ft: int,
        output_ft: int,
        kernel_size: int,
        non_linearity: nn.Module = nn.Identity()
    ):
        """Instantiate a synthesis residual layer.

        Args:
            input_ft (int): Input feature
            output_ft (int): Output feature
            kernel_size (int): Kernel size
            non_linearity (nn.Module): Non linear function applied at the very end
                of the forward. Defaults to nn.Identity()
        """
        super().__init__()

        assert input_ft == output_ft,\
            f'Residual layer in/out dim must match. Input = {input_ft}, output = {output_ft}'

        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        self.conv_layer = nn.Conv2d(
            input_ft,
            output_ft,
            kernel_size
        )

        self.non_linearity = non_linearity

        # More stable if a residual is initialized with all-zero parameters.
        # This avoids increasing the output dynamic at the initialization
        with torch.no_grad():
            self.conv_layer.weight.data = self.conv_layer.weight.data * 0.
            self.conv_layer.bias.data = self.conv_layer.bias.data * 0.

    def forward(self, x: Tensor) -> Tensor:
        return self.non_linearity(self.conv_layer(self.pad(x)) + x)
# =================== Conv layer layers for the Synthesis =================== #

