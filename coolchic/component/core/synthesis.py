# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import math
from typing import List, Literal, Optional, OrderedDict, Tuple

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
        self, in_channels: int, out_channels: int, kernel_size: int, residual: bool = False
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
            torch.empty(out_channels, in_channels // self.groups, kernel_size, kernel_size),
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
            self.weight = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        else:
            if self.weight.numel() == 0:
                self.weight = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
                return

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
    ``layers`` set the synthesis architecture. Each layer is described
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
    }

    possible_mode = ["linear", "residual"]

    def __init__(
        self,
        input_ft: int,
        layers: List[str],
        flag_linear_stabiliser: bool = True,
        flag_common_randomness: bool = False,
    ):
        """
        Args:
            input_ft: Number of input features :math:`C_{in}`. This corresponds
                to the number of latent features.
            layers: Description of each synthesis layer as a list of strings
                following the notation detailed above.
            flag_linear_stabiliser: True to add a linear stabiliser running parallel
                to the main trunk layers, as presented in the diagram below.
            flag_common_randomness: Set to true if half of the input features are common
                randomness features. In this case, the stabiliser layer does not take the
                common randomness features and as thus :math:`\\frac{C_{in}}{2}` input features.

        .. code-block:: none

                       ┌──────┐   ┌──────┐    ┌──────┐   ┌──────┐ trunk ┌─────┐
                x ──►──┤ Conv ├─►─┤ ReLU ├─►──┤ Conv ├─►─┤ ReLU ├───────┤  +  ├─► (mu, logscale)
                │      └──────┘   └──────┘    └──────┘   └──────┘       └─────┘
                ▼                                                          ▲
                │                      ┌─────┐                 stabiliser  │
                └──►───────────────────┤ Lin ├─────────────────────────────┘
                                       └─────┘
        """
        super().__init__()

        self.synth_branches = nn.ModuleList()
        self.input_ft = input_ft

        # Parse all the synthesis layer to get the number of output features
        # for the final synthesis layer
        self.output_ft = [int(lay.split("-")[0]) for lay in layers][-1]

        self.output_transform = SynthesisConv2d(self.output_ft, self.output_ft, 1, False)
        self.init_output_transform(None)
        # for param in self.output_transform.parameters():
        # param.requires_grad_(False)

        self.flag_linear_stabiliser = flag_linear_stabiliser
        self.flag_common_randomness = flag_common_randomness

        if self.flag_linear_stabiliser:
            self.n_input_ft_stabiliser = (
                self.input_ft if not self.flag_common_randomness else self.input_ft // 2
            )
            self.stabiliser_branch = SynthesisConv2d(
                self.n_input_ft_stabiliser,
                self.output_ft,
                kernel_size=1,
                residual=False,
            )
        else:
            self.stabiliser_branch = None

        main_branch_layers_list = nn.ModuleList()
        # Construct the hidden layer(s)
        for lay in layers:
            out_ft, k_size, mode, non_linearity = Synthesis._parse_layer_syntax(lay)

            # Check that mode and non linearity is correct
            assert mode in Synthesis.possible_mode, (
                f"Unknown mode. Found {mode}. Should be in {Synthesis.possible_mode}"
            )

            assert non_linearity in Synthesis.possible_non_linearity, (
                f"Unknown non linearity. Found {non_linearity}. "
                f"Should be in {Synthesis.possible_non_linearity.keys()}"
            )

            # Instantiate them
            main_branch_layers_list.append(
                SynthesisConv2d(input_ft, out_ft, k_size, residual=mode == "residual")
            )
            main_branch_layers_list.append(Synthesis.possible_non_linearity[non_linearity]())

            input_ft = out_ft

        self.main_branch = nn.Sequential(*main_branch_layers_list)

    @classmethod
    def _parse_layer_syntax(cls, layer_description: str) -> Tuple[int, int, str, str]:
        """Parse a string description of a synthesis layer and return the number of output
        features, the kernel size, the mode (normal or residual) and the non-linearity
        (none or relu).

        Args:
            layer_description: String description the layer. Format:
                <out_features>-<kernel_size>-<mode>-<non_linearity>

        Returns:
            Tuple[int, int, str, str]: out_feature, kernel_size, mode, non_linearity
        """
        out_ft, k_size, mode, non_linearity = layer_description.split("-")
        out_ft = int(out_ft)
        k_size = int(k_size)

        # Check that mode and non linearity is correct
        if mode not in Synthesis.possible_mode:
            raise ValueError(f"Unknown mode. Found {mode}. Should be in {Synthesis.possible_mode}")

        if non_linearity not in Synthesis.possible_non_linearity:
            raise ValueError(
                f"Unknown non linearity. Found {non_linearity}. "
                f"Should be in {Synthesis.possible_non_linearity.keys()}"
            )

        return out_ft, k_size, mode, non_linearity

    def forward(self, x: Tensor) -> Tensor:
        """Perform the synthesis forward pass :math:`\hat{\mathbf{x}} =
        f_{\\theta}(\hat{\mathbf{z}})`, where :math:`\hat{\mathbf{x}}` is the
        :math:`(B, C_{out}, H, W)` synthesis output, :math:`\hat{\mathbf{z}}` is
        the :math:`(B, C_{in}, H, W)` synthesis input (i.e. the upsampled latent
        variable) and :math:`\\theta` the synthesis parameters.

        Args:
            x: Dense latent representation :math:`(B, C_{in}, H, W)`.

        Returns:
            Raw output features :math:`(B, C_{out}, H, W)`.
        """

        if self.flag_linear_stabiliser:
            x = self.main_branch(x) + self.stabiliser_branch(
                x[:, : self.n_input_ft_stabiliser, :, :]
            )

        else:
            x = self.main_branch(x)

        return self.output_transform(x)

    def get_param(
        self, which: Optional[Literal["weight", "bias"]] = None
    ) -> OrderedDict[str, Tensor]:
        """Return **a copy** of the weights and biases inside the module.

        Args:
            which (Optional[Literal["weight", "bias"]]): Wether to return only the weights
                 or the biases. If None, return everything. Defaults to None.

        Returns:
            A copy of all weights & biases in the layers.
        """
        # Detach & clone to create a copy
        param = OrderedDict(
            {
                param_name: param_value.detach().clone()
                for param_name, param_value in self.named_parameters()
            }
        )

        if which is not None:
            available_filters = ["weight", "bias"]
            if which not in available_filters:
                raise ValueError(
                    f"get_param() which should be in {available_filters} or None "
                    f"to get all parameters Found which={which}"
                )

            param = {
                param_name: param_value
                for param_name, param_value in param.items()
                if which in param_name
            }

        return param

    def set_param(self, param: OrderedDict[str, Tensor]):
        """Replace the current parameters of the module with param.

        Args:
            param: Parameters to be set.
        """
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
        """Re-initialize in place the params of all the ``SynthesisConv2d`` layers."""
        for layer in self.main_branch.children():
            if isinstance(layer, SynthesisConv2d):
                layer.initialize_parameters()

        for layer in self.stabiliser_branch.children():
            if isinstance(layer, SynthesisConv2d):
                layer.initialize_parameters()

    @torch.no_grad()
    def init_output_transform(self, img_min_max: Optional[Tensor] = None) -> None:

        weight_shape = (self.output_ft, self.output_ft, 1, 1)
        bias_shape = self.output_ft

        if img_min_max is None:
            weight = torch.eye(self.output_ft).view(weight_shape)
            bias = torch.zeros(bias_shape)

        else:
            img_min = img_min_max[:, 0]
            img_max = img_min_max[:, 1]

            weight = torch.diag(img_max - img_min).view(weight_shape)
            bias = img_min

        self.output_transform.weight = nn.Parameter(weight, requires_grad=False)
        self.output_transform.bias = nn.Parameter(bias, requires_grad=False)
