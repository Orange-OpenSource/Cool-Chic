# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
from torch import Tensor, nn
from typing import List

from models.quantizable_module import QuantizableModule
from utils.constants import POSSIBLE_Q_STEP_SYN_NN

class SynthesisLayer(nn.Module):
    def __init__(self, input_ft: int, output_ft: int, kernel_size: int):
        """Instantiate a synthesis layer.

        Args:
            input_ft (int): Input feature
            output_ft (int): Output feature
            kernel_size (int): Kernel size
        """
        super().__init__()

        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        self.conv_layer = nn.Conv2d(
            input_ft,
            output_ft,
            kernel_size
        )

        # More stable if initialized as a zero-bias layer with smaller variance
        # for the weights.
        with torch.no_grad():
            self.conv_layer.weight.data = self.conv_layer.weight.data / output_ft ** 2
            self.conv_layer.bias.data = self.conv_layer.bias.data * 0.

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_layer(self.pad(x))


class SynthesisResidualLayer(nn.Module):
    def __init__(self, input_ft: int, output_ft: int, kernel_size: int):
        """Instantiate a synthesis residual layer.

        Args:
            input_ft (int): Input feature
            output_ft (int): Output feature
            kernel_size (int): Kernel size
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

        # More stable if a residual is initialized with all-zero parameters.
        # This avoids increasing the output dynamic at the initialization
        with torch.no_grad():
            self.conv_layer.weight.data = self.conv_layer.weight.data * 0.
            self.conv_layer.bias.data = self.conv_layer.bias.data * 0.

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_layer(self.pad(x)) + x


class SynthesisAttentionLayer(nn.Module):
    def __init__(self, input_ft: int, output_ft: int, kernel_size: int):
        """Instantiate a synthesis attention layer.

        Args:
            input_ft (int): Input feature
            output_ft (int): Output feature
            kernel_size (int): Kernel size
        """
        super().__init__()

        assert input_ft == output_ft,\
            f'Attention layer in/out dim must match. Input = {input_ft}, output = {output_ft}'

        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))

        self.conv_layer_trunk = nn.Conv2d(
            input_ft,
            output_ft,
            kernel_size
        )
        self.conv_layer_sigmoid = nn.Conv2d(
            input_ft,
            output_ft,
            kernel_size
        )

        # Trunk is initialized as a residual block (i.e. all zero parameters)
        # Sigmoid branch is initialized as a linear layer (i.e. no bias and smaller variance
        # for the weights)
        with torch.no_grad():
            self.conv_layer_trunk.weight.data = self.conv_layer_trunk.weight.data * 0.
            self.conv_layer_trunk.bias.data = self.conv_layer_trunk.bias.data * 0.
            self.conv_layer_sigmoid.weight.data = self.conv_layer_sigmoid.weight.data / output_ft ** 2

    def forward(self, x: Tensor) -> Tensor:
        trunk = self.conv_layer_trunk(self.pad(x))
        weight = torch.sigmoid(self.conv_layer_sigmoid(self.pad(x)))
        return trunk * weight + x


class Synthesis(QuantizableModule):
    possible_non_linearity = {
        'none': nn.Identity,
        'relu': nn.ReLU,
        'leakyrelu': nn.LeakyReLU,
    }
    possible_mode = {
        'linear': SynthesisLayer,
        'residual': SynthesisResidualLayer,
        'attention': SynthesisAttentionLayer,
    }

    def __init__(self, input_ft: int, layers_dim: List[str]):
        super().__init__(possible_q_steps=POSSIBLE_Q_STEP_SYN_NN)
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for layers in layers_dim:
            out_ft, k_size, mode, non_linearity = layers.split('-')
            out_ft = int(out_ft)
            k_size = int(k_size)

            # Check that mode and non linearity is correct
            assert mode in Synthesis.possible_mode,\
                f'Unknown mode. Found {mode}. Should be in {Synthesis.possible_mode.keys()}'

            assert non_linearity in Synthesis.possible_non_linearity,\
                f'Unknown non linearity. Found {non_linearity}. '\
                f'Should be in {Synthesis.possible_non_linearity.keys()}'

            # Instantiate them
            layers_list.append(Synthesis.possible_mode[mode](input_ft, out_ft, k_size))
            layers_list.append(Synthesis.possible_non_linearity[non_linearity]())

            input_ft = out_ft

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class NoiseQuantizer(torch.autograd.Function):
    noise_derivative: float = 100.

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        y = x + (torch.rand_like(x) - 0.5)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out * NoiseQuantizer.noise_derivative


class STEQuantizer(torch.autograd.Function):
    ste_derivative: float = 1e-2

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        y = torch.round(x)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out * STEQuantizer.ste_derivative

def quantize(x: Tensor, training: bool) -> Tensor:
    """Quantize a tensor with a unitary quantization step

    Args:
        x (Tensor): Tensor to be quantized
        training (bool): True if we're training. In this case we use the
            additive noise model. Otherwise, the actual quantization (round)
            is used
        log_2_gain (Tensor): Tensor of shape [1] containing the quantization gain.

    Returns:
        Tensor: The quantized version of x.
    """
    return x + (torch.rand_like(x) - 0.5) if training else torch.round(x)
