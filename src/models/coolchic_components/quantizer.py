# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import math
import torch
from torch import nn, Tensor


def softround(x: Tensor, t: float) -> Tensor:
    """Apply the soft round function with temperature t on a tensor x.

    Args:
        x (Tensor): Input tensor of any size
        t (float): Soft round temperature

    Returns:
        Tensor: Soft-rounded tensor
    """
    delta = x - torch.floor(x) - 0.5
    return torch.floor(x) + 0.5 * torch.tanh(delta / t) / math.tanh(1 / (2 * t)) + 0.5


class NoiseQuantizer(nn.Module):
    def __init__(self, soft_round_temperature: float = 0.3, kumaraswamy_param: float = 1.0):
        """Initialize the proxy for the actual quantization. The noise quantizer follows the
        operations described in [1] namely:
            1. Use a soft round function instead of the non-differentiable round function
            2. Add a kumaraswamy noise to prevent the network from learning the inverse softround function
            3. Re-apply the soft round function as advocated by [2]

            The soft round is parameterized by a temperature, where temperature = 0 corresponds to the
        actual rounding function while an infinite temperature corresponds to the identity function.
        The kumaraswamy noise is parameterized by a <kumaraswamy_param> in [1., + inf]. Setting it to 1
        corresponds to a uniform distribution

        [1] "C3: High-performance and low-complexity neural compression from a single image or video", Kim et al.
        [2] "Universally Quantized Neural Compression", Agustsson et al.

        Args:
            soft_round_temperature (float, optional): Soft round temperature. Defaults to 0.3.
            kumaraswamy_param (float, optional): Kumaraswamy noise distribution parameter. Defaults to 1.0.
        """
        super().__init__()
        self.soft_round_temperature = soft_round_temperature
        self.kumaraswamy_param = kumaraswamy_param

    def forward(self, x: Tensor) -> Tensor:
        """Apply the noise-based quantizer on a tensor x.

        Args:
            x (Tensor): Tensor to be quantized of any size

        Returns:
            Tensor: Quantized tensor
        """
        noise = self.generate_kumaraswamy_noise(torch.rand_like(x), self.kumaraswamy_param)
        y = softround(softround(x, self.soft_round_temperature) + (noise), self.soft_round_temperature)
        return y

    def generate_kumaraswamy_noise(self, uniform_noise: Tensor, kumaraswamy_param: float) -> Tensor:
        """Reparameterize a uniform noise in [0. 1.] as a kumaraswamy noise shifted to [-0.5, 0.5]

        Args:
            uniform_noise (Tensor): A uniform noise in [0., 1.] with any size.
            kumaraswamy_param (float): Hyperparameter. Set to 1 for a uniform noise

        Returns:
            Tensor: A kumaraswamy noise with the same size in [-0.5, 0.5], parameterized by self.temp
        """
        # This relation between a and b allows to always have a mode of 0.5
        a = kumaraswamy_param
        b = (2 ** a * (a - 1) + 1) / a

        # Use the inverse of the repartition function to sample a kumaraswamy noise in [0., 1.]
        kumaraswamy_noise = (1 - (1 - uniform_noise) ** (1 / b)) ** (1 / a)

        # Shift the noise to have it in [-0.5, 0.5]
        return kumaraswamy_noise - 0.5


class STEQuantizer(nn.Module):
    def __init__(self, soft_round_temperature: float = 1e-4):
        """Badly named Straight-through quantizer. Forward pass is the actual quantization,
        backward pass is the backward of the softround function.

        Args:
            soft_round_temperature (float, optional): Temperature of the soft round function
                for the backward pass. Defaults to 1e-4.
        """
        super().__init__()
        self.soft_round_temperature = soft_round_temperature

    def forward(self, x: Tensor) -> Tensor:
        """Quantize a tensor x. Gradient is set to the gradient of the soft round function

        Args:
            x (Tensor): Tensor to be quantized with any size.

        Returns:
            Tensor: Quantized tensor.
        """
        # From the forward point of view (i.e. entering into the torch.no_grad()), we have
        # y = softround(x) - softround(x) + round(x) = round(x). From the backward point of view
        # we have y = softround(x) meaning that dy / dx = d softround(x) / dx.
        y = softround(x, self.soft_round_temperature)
        with torch.no_grad():
            y = y - softround(x, self.soft_round_temperature) + torch.round(x)
        return y

