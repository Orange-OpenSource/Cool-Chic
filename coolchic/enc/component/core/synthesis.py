# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import math
from typing import List, OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SynthesisConv2d(nn.Module):
    """Instantiate a synthesis layer applying the following operation to an
    input tensor :math:`\\mathbf{x}` with shape :math:`[B, C_{in}, H, W]`, producing
    an output tensor :math:`\\mathbf{y}` with shape :math:`[B, C_{out}, H, W]`.

    .. math::

        \\mathbf{y} =
        \\begin{cases}
            \mathrm{conv}(\\mathbf{x}) + \\mathbf{x} & \\text{if residual,} \\\\
            \mathrm{conv}(\\mathbf{x}) & \\text{otherwise.} \\\\
        \\end{cases}
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        residual: bool = False,
    ):
        """
        Args:
            in_channels: Number of input channels :math:`C_{in}`.
            out_channels: Number of output channels :math:`C_{out}`.
            kernel_size: Kernel size (height and width are identical)
            residual: True to add a residual connection to the layer.
                Default to False.
        """
        super().__init__()

        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = int((kernel_size - 1) / 2)

        # -------- Instantiate empty parameters, set by the initialize function
        self.groups = 1  # Hardcoded for now
        self.weight = nn.Parameter(
            torch.empty(
                out_channels, in_channels // self.groups, kernel_size, kernel_size
            ),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.empty((out_channels)), requires_grad=True)
        self.initialize_parameters()
        # -------- Instantiate empty parameters, set by the initialize function

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of this layer.

        Args:
            x: Input tensor of shape :math:`[B, C_{in}, H, W]`.

        Returns:
            Output tensor of shape :math:`[B, C_{out}, H, W]`.
        """
        padded_x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
        y = F.conv2d(padded_x, self.weight, self.bias, groups=self.groups)

        if self.residual:
            return y + x
        else:
            return y

    def initialize_parameters(self) -> None:
        """Initialize **in place** the weights and biases of the
        ``SynthesisConv2d`` layer.

        * Biases are always set to zero.

        * Weights are set to zero if ``residual`` is ``True``. Otherwise, they
          follow a Uniform distribution: :math:`\\mathbf{W} \sim
          \\mathcal{U}(-a, a)`, where :math:`a =
          \\frac{1}{C_{out}^2\\sqrt{C_{in}k^2}}` with :math:`k` the kernel size.
        """
        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)

        if self.residual:
            self.weight = nn.Parameter(
                torch.zeros_like(self.weight), requires_grad=True
            )
        else:
            # Default PyTorch initialization for convolution 2d: weight ~ Uniform(-sqrt(k), sqrt(k))
            # Empirically, it works better if we further divide the resulting weights by output_ft ** 2
            out_channel, in_channel_divided_by_group, kernel_height, kernel_weight = (
                self.weight.size()
            )
            in_channel = in_channel_divided_by_group * self.groups
            k = self.groups / (in_channel * kernel_height * kernel_weight)
            sqrt_k = math.sqrt(k)

            self.weight = nn.Parameter(
                (torch.rand_like(self.weight) - 0.5) * 2 * sqrt_k / (out_channel**2),
                requires_grad=True,
            )


class Synthesis(nn.Module):
    """Instantiate Cool-chic convolution-based synthesis transform. It
    performs the following operation.

    .. math::

        \hat{\mathbf{x}} = f_{\\theta}(\hat{\mathbf{z}}).

    Where :math:`\hat{\mathbf{x}}` is the :math:`[B, C_{out}, H, W]`
    synthesis output, :math:`\hat{\mathbf{z}}` is the :math:`[B, C_{in}, H,
    W]` synthesis input (i.e. the upsampled latent variable) and
    :math:`\\theta` the synthesis parameters.

    The synthesis is composed of one or more convolution layers,
    instantiated using the class ``SynthesisConv2d``. The parameter
    ``layers_dim`` set the synthesis architecture. Each layer is described
    as follows: ``<output_dim>-<kernel_size>-<type>-<non_linearity>``

    * ``output_dim``: number of output features :math:`C_{out}`.

    * ``kernel_size``: spatial dimension of the kernel. Use 1 to mimic an MLP.

    * ``type``: either ``linear`` or ``residual`` *i.e.*

        .. math::

            \\mathbf{y} =
            \\begin{cases}
                \mathrm{conv}(\\mathbf{x}) + \\mathbf{x} & \\text{if residual,} \\\\
                \mathrm{conv}(\\mathbf{x}) & \\text{otherwise.} \\\\
            \\end{cases}

    * ``non_linearity``: either ``none`` (no non-linearity) or ``relu``.
        The non-linearity is applied after the residual connection if any.

    Example of a convolution layer with 40 input features, 3 output features, a
    residual connection followed with a relu: ``40-3-residual-relu``

    """
    possible_non_linearity = {
        "none": nn.Identity,
        "relu": nn.ReLU,
        # "leakyrelu": nn.LeakyReLU,    # Unsupported by the decoder
        # "gelu": nn.GELU,              # Unsupported by the decoder
    }

    possible_mode = ["linear", "residual"]

    def __init__(self, input_ft: int, layers_dim: List[str]):
        """
        Args:
            input_ft: Number of input features :math:`C_{in}`. This corresponds
                to the number of latent features.
            layers_dim: Description of each synthesis layer as a list of strings
                following the notation detailed above.
        """
        super().__init__()

        self.synth_branches = nn.ModuleList()
        self.input_ft = input_ft
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for layers in layers_dim:
            out_ft, k_size, mode, non_linearity = layers.split("-")
            out_ft = int(out_ft)
            k_size = int(k_size)

            # Check that mode and non linearity is correct
            assert (
                mode in Synthesis.possible_mode
            ), f"Unknown mode. Found {mode}. Should be in {Synthesis.possible_mode}"

            assert non_linearity in Synthesis.possible_non_linearity, (
                f"Unknown non linearity. Found {non_linearity}. "
                f"Should be in {Synthesis.possible_non_linearity.keys()}"
            )

            # Instantiate them
            layers_list.append(
                SynthesisConv2d(input_ft, out_ft, k_size, residual=mode == "residual")
            )
            layers_list.append(Synthesis.possible_non_linearity[non_linearity]())

            input_ft = out_ft

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the synthesis forward pass :math:`\hat{\mathbf{x}} =
        f_{\\theta}(\hat{\mathbf{z}})`, where :math:`\hat{\mathbf{x}}` is the
        :math:`[B, C_{out}, H, W]` synthesis output, :math:`\hat{\mathbf{z}}` is
        the :math:`[B, C_{in}, H, W]` synthesis input (i.e. the upsampled latent
        variable) and :math:`\\theta` the synthesis parameters.

        Args:
            x: Dense latent representation :math:`[B, C_{in}, H, W]`.

        Returns:
            Raw output features :math:`[B, C_{out}, H, W]`.
        """
        return self.layers(x)

    def partial_forward(self, x: Tensor, last_layer_idx: int = 1) -> Tensor:
        """Perform a "partial" forward of the synthesis, i.e. stopping at the
        <last_layer_idx>-th layer. We do not count non-linearity as actual
        layer. That is:

        x --> Conv --> ReLU --> Conv --> Conv --> ReLU --> out
                             ^        ^                 ^
        last_layer_idx       1        2                 3  
        

        Args:
            x: Dense latent representation :math:`[B, C_{in}, H, W]`.
            last_layer_idx: Last layer index. Defaults to 1.

        Returns:
            Features [B, C, H, W]`.
        """
        i = 0
        layer_idx = 0
        while True:
            if isinstance(self.layers[i], SynthesisConv2d):
                layer_idx += 1

            if layer_idx > last_layer_idx:
                return x

            x = self.layers[i](x)
            i += 1


    def get_param(self) -> OrderedDict[str, Tensor]:
        """Return **a copy** of the weights and biases inside the module.

        Returns:
            A copy of all weights & biases in the layers.
        """
        # Detach & clone to create a copy
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]):
        """Replace the current parameters of the module with param.

        Args:
            param: Parameters to be set.
        """
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
        """Re-initialize in place the params of all the ``SynthesisConv2d`` layers."""
        for layer in self.layers.children():
            if isinstance(layer, SynthesisConv2d):
                layer.initialize_parameters()
