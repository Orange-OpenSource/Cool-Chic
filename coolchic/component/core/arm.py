# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from typing import List, Literal, Optional, OrderedDict, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, index_select, nn

# no scale smaller than exp(-5) = 6e-3 or bigger than exp(5) = 150
LOG_SCALE_MIN = -5
LOG_SCALE_MAX = 5


class ArmLinear(nn.Module):
    """Create a Linear layer of the Auto-Regressive Module (ARM). This is a
    wrapper around the usual ``nn.Linear`` layer of PyTorch, with a custom
    initialization. It performs the following operations:

    * :math:`\\mathbf{x}_{out} = \\mathbf{W}\\mathbf{x}_{in} + \\mathbf{b}` if
      ``residual`` is ``False``

    * :math:`\\mathbf{x}_{out} = \\mathbf{W}\\mathbf{x}_{in} + \\mathbf{b} +
      \\mathbf{x}_{in}` if ``residual`` is ``True``.

    The input  :math:`\\mathbf{x}_{in}` is a :math:`[B, C_{in}]` tensor, the
    output :math:`\\mathbf{x}_{out}` is a :math:`[B, C_{out}]` tensor.

    The layer weight and bias shapes are :math:`\\mathbf{W} \\in
    \\mathbb{R}^{C_{out} \\times C_{in}}` and :math:`\\mathbf{b} \\in
    \\mathbb{R}^{C_{out}}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = False,
    ):
        """
        Args:
            in_channels: Number of input features :math:`C_{in}`.
            out_channels: Number of output features :math:`C_{out}`.
            residual: True to add a residual connection to the layer. Defaults to False.
        """

        super().__init__()

        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        # -------- Instantiate empty parameters, set by the initialize function
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.empty((out_channels)), requires_grad=True)
        self.initialize_parameters()
        # -------- Instantiate empty parameters, set by the initialize function

    def initialize_parameters(self) -> None:
        """Initialize **in place** the weight and the bias of the linear layer.

        * Biases are always set to zero.

        * Weights are set to zero if ``residual == True``. Otherwise, sample
          from the Normal distribution: :math:`\\mathbf{W} \\sim \\mathcal{N}(0,
          \\tfrac{1}{C_{out}^4})`.
        """
        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)
        if self.residual:
            self.weight = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        else:
            out_channel = self.weight.size()[0]
            self.weight = nn.Parameter(
                torch.randn_like(self.weight) / out_channel**2, requires_grad=True
            )

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of this layer.

        Args:
            x: Input tensor of shape :math:`[B, C_{in}]`.

        Returns:
            Tensor with shape :math:`[B, C_{out}]`.
        """
        if self.residual:
            return F.linear(x, self.weight, bias=self.bias) + x

        # Not residual
        else:
            return F.linear(x, self.weight, bias=self.bias)


class Arm(nn.Module):
    """Autoregressive probability module, modelling the
    conditional distribution :math:`p_{\\psi}(\\hat{y}_i \\mid \\mathbf{s}_i,
    \\mathbf{f}_i)` of a (quantized) latent pixel :math:`\\hat{y}_i`,
    conditioned on neighboring already decoded context pixels. These context
    pixels are either causal spatial neighbors :math:`\\mathbf{s}_i`, extracted
    from the same latent grid, or inter-feature context :math:`\\mathbf{f}_i`
    extracted thanks to an IFCE module from already decoded.

    The distribution :math:`p_{\\psi}` is assumed to follow a Laplace
    distribution, parameterized by an expectation :math:`\\mu` and a scale
    :math:`b`, where the scale and the variance :math:`\\sigma^2` are related as
    follows :math:`\\sigma^2 = 2 b ^2`.

    The parameters of the Laplace distribution for a given latent pixel
    :math:`\\hat{y}_i` are obtained by passing the context pixel through an MLP
    :math:`f_{\\psi}`:

    .. math::

        p_{\\psi}(\\hat{y}_i \\mid \\mathbf{c}_i) \\sim \\mathcal{L}(\\mu_i, b_i),
        \\text{ where } \\mu_i, b_i =
        f_{\\psi}(\\mathtt{concat}(\\mathbf{s}_i,\\mathbf{f}_i)).

    .. attention::

        The MLP :math:`f_{\\psi}` has a few constraint on its architecture:

        * The width of all hidden layers (i.e. the output of all layers except
          the final one) are identical to the number of pixel contexts;

        * All layers except the last one are residual layers, followed by a
          ``ReLU`` non-linearity;

    The MLP :math:`f_{\\psi}` is made of custom Linear layers instantiated from
    the ``ArmLinear`` class.
    """

    def __init__(
        self,
        dim_arm: int,
        n_hidden_layers_arm: int,
        n_out_features: int = 2,
        flag_linear_stabiliser: bool = True,
    ):
        """
        Args:
            dim_arm: Number of context pixels **and** dimension of all hidden layers.
            n_hidden_layers_arm: Number of hidden layers. Set it to 0 for a linear ARM.
            n_out_features: Number of output features. Should usually be 2 for the expecation
                :math:`\\mu` and scale :math:`b`.
            flag_linear_stabiliser: True to add a linear stabiliser running parallel
                to the main trunk layers, as presented in the diagram below:

        .. code-block:: none

                       ┌─────┐   ┌──────┐    ┌─────┐   ┌──────┐ trunk ┌─────┐
                x ──►──┤ Lin ├─►─┤ ReLU ├─►──┤ Lin ├─►─┤ ReLU ├───────┤  +  ├─► (mu, logscale)
                │      └─────┘   └──────┘    └─────┘   └──────┘       └─────┘
                ▼                                                        ▲
                │                      ┌─────┐               stabiliser  │
                └──►───────────────────┤ Lin ├───────────────────────────┘
                                       └─────┘
        """
        super().__init__()

        self.dim_arm = dim_arm
        self.n_hidden_layers_arm = n_hidden_layers_arm
        self.n_out_features = n_out_features

        # This will be subtracted to one of the output feature of the ARM, the
        # one corresponding to the scale.
        self.register_buffer("log_shift", torch.tensor(-4), persistent=False)

        # ======================== Construct the MLP ======================== #
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for i in range(n_hidden_layers_arm):
            layers_list.append(ArmLinear(dim_arm, dim_arm, residual=True))
            layers_list.append(nn.ReLU())

        # Construct the output layer. It always has 2 outputs (mu and scale)
        layers_list.append(ArmLinear(dim_arm, self.n_out_features, residual=False))
        self.mlp = nn.Sequential(*layers_list)

        self.flag_linear_stabiliser = flag_linear_stabiliser

        if self.flag_linear_stabiliser:
            self.stabiliser_branch = ArmLinear(
                self.dim_arm,
                self.n_out_features,
            )
        else:
            self.stabiliser_branch = None

        # ======================== Construct the MLP ======================== #

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform the auto-regressive module (ARM) forward pass. The ARM takes
        as input a tensor of shape :math:`(B, C_{in})` i.e. :math:`B` contexts
        with :math:`C` values each. ARM outputs :math:`(B, C_{out})`.

        Usually, :math:`C_{out} = 2` *i.e.,* two values per pixel describing the
        expectation and scale of the Laplace distribution. The function
        :code:`reparameterize_input` transforms these quantities into proper
        expectation and scale.

        .. warning::

            Note that the ARM expects input to be flattened i.e. spatial
            dimensions :math:`H, W` are collapsed into a single batch-like
            dimension :math:`B = HW`, leading to an input of shape :math:`(B,
            C)`, gathering the :math:`C` contexts for each of the :math:`B`
            pixels to model.

        Args:
            x: Concatenation of all input contexts
                :math:`\\mathbf{c}_i`. Tensor of shape :math:`(B, C_{in})`.

        Returns:
            Concatenation of all output quantities derived from the input contexts.
                Tensor of shape :math:`(B, C_{out})`.
        """
        if self.flag_linear_stabiliser:
            return self.mlp(x) + self.stabiliser_branch(x)

        else:
            return self.mlp(x)

    def reparameterize_output(self, raw_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Reparameterize the raw output of the :math:`(B, 2) ARM into mu and scale parameters.

        The expectation :math:`\\mu` is left unchanged from the ARM output.
        The scale goes through an exponential reparameterization: :math:`b = e^{(x - 4)}`

        Args:
            x: Raw ARM output. Shape is :math:`(B, 2)`.

        Returns:
            Tuple[Tensor, Tensor]. Mu and scale parameters an identical shape of :math:`(B)` elements.
        """
        if raw_output.size()[1] != 2 or len(raw_output.size()) != 2:
            raise ValueError(f"ARM output should have dimension [B, 2]. Found {raw_output.size()}")

        mu = raw_output[:, 0]
        log_scale = raw_output[:, 1]

        scale = torch.exp(
            torch.clamp(log_scale + self.log_shift, min=LOG_SCALE_MIN, max=LOG_SCALE_MAX)
        )

        return mu, scale

    def get_param(
        self, which: Optional[Literal["weight", "bias"]] = None, detach_and_clone: bool = True
    ) -> OrderedDict[str, Tensor]:
        """Return the weights and biases inside the module.

        Args:
            which (Optional[Literal["weight", "bias"]]): Wether to return only the weights
                 or the biases. If None, return everything. Defaults to None.
            detach_and_clone (bool): If True, return a copy of the detached parameters.
                Defaults to True

        Returns:
            A dict of all the required parameters in the layers.
        """
        # Detach & clone to create a copy
        param = OrderedDict(
            {
                param_name: param_value.detach().clone() if detach_and_clone else param_value
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

    def set_param(self, param: OrderedDict[str, Tensor], strict: bool = True) -> None:
        """Replace the current parameters of the module with param.

        Args:
            param: Parameters to be set.
        """
        self.load_state_dict(param, strict=strict)

    def reinitialize_parameters(self) -> None:
        """Re-initialize in place the parameters of all the ArmLinear layers."""
        for layer in self.mlp.children():
            if isinstance(layer, ArmLinear):
                layer.initialize_parameters()


class Ifce(nn.Module):
    """Inter Feature Context Extractor (IFCE) contains all the IFCE
    :math:`f_{\\chi^(k)}`, each of them dedicated to the :math:`k`-th latent
    grid.

    The role of each IFCE :math:`f_{\\chi^(k)}` is to compute for each pixel of the
    :math:`k`-th latent grid a context vector of :math:`C_f` elements based on the already
    decoded latent grids.
    """

    def __init__(self, input_features_ifce: List[int], output_features_ifce: int):
        """
        Args:
            input_features_ifce: Number of input features for each of the IFCE,
                one per latent grid. For instance
                :code:`input_features_ifce=[3,2,0,0]` indicates that the first feature
                (highest resolution) uses the 3 already decoded features as context, while
                the second feature uses the 2 already decoded features as context. 0 indicates
                that no IFCE is used for the current feature.
            output_features_ifce: How many elements :math:`C_f` are computed from the raw context values.
        """
        super().__init__()

        self.arms = nn.ModuleList()
        self.index_to_arm = dict()
        self.output_features_ifce = output_features_ifce
        self.input_features_ifce = input_features_ifce
        internal_index = 0
        for i, input_ft_i in enumerate(self.input_features_ifce):
            # No IFCE when we don't have any output features
            if input_ft_i == 0:
                continue

            self.arms.append(
                Arm(
                    dim_arm=input_ft_i,
                    n_hidden_layers_arm=0,
                    n_out_features=self.output_features_ifce,
                    flag_linear_stabiliser=False,
                )
            )
            self.index_to_arm[i] = internal_index
            internal_index += 1

    def forward(self, x: Tensor, latent_grid_idx: int) -> Tensor:
        """From a raw values extracted from already decoded latent grids :math:`\\mathbf{r}`,
        compute a feature context :math:`\\mathbf{f} = f_{\\chi^(k)}(\\mathbf{r})`.

        Args:
            x (Tensor): Raw values extracted from already decoded latent grids :math:`\\mathbf{r}`
                Shape is :math:`(B, C_{in}^{(i)})`, with :math:`C_{in}^{(i)}` the :math:`i`-th element
                in the :code:`input_features_ifce` list from the :code:`__init__` function.
            latent_grid_idx (int): Index of the IFCE :math:`k` (and of the assocaited latent grids).

        Returns:
            Tensor: Transformed context :math:`\\mathbf{f}`. Shape is :math:`(B, C_f)`
        """
        return self.arms[self.index_to_arm[latent_grid_idx]](x)

    def get_param(
        self, which: Optional[Literal["weight", "bias"]] = None, detach_and_clone: bool = True
    ) -> OrderedDict[str, Tensor]:
        """Return the weights and biases inside the module.

        Args:
            which (Optional[Literal["weight", "bias"]]): Wether to return only the weights
                 or the biases. If None, return everything. Defaults to None.
            detach_and_clone (bool): If True, return a copy of the detached parameters.
                Defaults to True

        Returns:
            A dict of all the required parameters in the layers.
        """
        # Detach & clone to create a copy
        param = OrderedDict(
            {
                param_name: param_value.detach().clone() if detach_and_clone else param_value
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

    def set_param(self, param: OrderedDict[str, Tensor], strict: bool = True) -> None:
        """Replace the current parameters of the module with param.

        Args:
            param: Parameters to be set.
        """
        self.load_state_dict(param, strict=strict)

    def reinitialize_parameters(self) -> None:
        """Re-initialize in place the parameters of all the ArmLinear layer."""
        for layer in self.mlp.children():
            if isinstance(layer, ArmLinear):
                layer.initialize_parameters()


def _get_neighbor(x: Tensor, non_zero_pixel_ctx_idx: Tensor, mask_size: int) -> Tensor:
    """Use the unfold function to extract the neighbors of each pixel in x.

    Args:
        x: [1, C, H, W] feature map from which we wish to extract the
            neighbors
        non_zero_pixel_ctx_idx (Tensor): [N] 1D tensor containing the indices
            of the non zero context pixels (i.e. floor(N ** 2 / 2) - 1).
            It looks like: [0, 1, ..., floor(N ** 2 / 2) - 1].
            This allows to use the index_select function, which is significantly
            faster than usual indexing.

    Returns:
        torch.tensor: [H * W, C, floor(N ** 2 / 2) - 1] the spatial neighbors
            the floor(N ** 2 / 2) - 1 neighbors of each H * W pixels and C channels.
    """
    pad = int((mask_size - 1) / 2)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)

    # Shape of x_unfold is [B, C, H, W, mask_size, mask_size] --> [B * C * H * W, mask_size * mask_size]
    # reshape is faster than einops.rearrange
    x_unfold = x_pad.unfold(2, mask_size, step=1).unfold(3, mask_size, step=1)
    x_unfold = rearrange(x_unfold, "b c h w mask_h mask_w -> (b h w) c (mask_h mask_w)", b=1)
    # Select the pixels for which the mask is not zero
    neighbor = index_select(x_unfold, dim=2, index=non_zero_pixel_ctx_idx)
    return neighbor


def _laplace_cdf(x: Tensor, expectation: Tensor, scale: Tensor) -> Tensor:
    """Compute the laplace cumulative evaluated in x. All parameters
    must have the same dimension.
    Re-implemented here coz it is faster than calling the Laplace distribution
    from torch.distributions.

    Args:
        x (Tensor): Where the cumulative if evaluated.
        expectation (Tensor): Expectation.
        scale (Tensor): Scale

    Returns:
        Tensor: CDF(x, mu, scale)
    """
    shifted_x = x - expectation
    return 0.5 - 0.5 * (shifted_x).sign() * torch.expm1(-(shifted_x).abs() / scale)


def compute_rate(x: Tensor, expectation: Tensor, scale: Tensor) -> Tensor:
    """Return the per-symbol rate of the tensor x, when entropy coded with a Laplace distribution
    with per-symbol expectation and scale parameters

    All inputs (x, expectation scale) and the output have the same shape

    Args:
        x (Tensor): Data whose rate will be measured.
        expectation (Tensor): Expectation of the Laplace distribution
        scale (Tensor): Scale of the Laplace distribution

    Returns:
        Tensor: Per-symbol rate (in bits).
    """
    # Compute the rate (i.e. the entropy of flat latent knowing mu and scale)
    proba = torch.clamp_min(
        _laplace_cdf(x + 0.5, expectation, scale) - _laplace_cdf(x - 0.5, expectation, scale),
        min=2**-16,  # No value can cost more than 16 bits.
    )
    return -torch.log2(proba)


# -------------------------------------------------------------- #
# -------------------                        ------------------- #
# -------------------  AUTO-REGRESSIVE MASK  ------------------- #
# -------------------                        ------------------- #
# -------------------------------------------------------------- #
MAX_ARM_MASK_SIZE = 9


def get_priority_order() -> Tensor:
    """Order in which we'll use the neighboring pixels. When n_spatial_ctx = N,
    we use the neighbors with priority in [0, N - 1]
    """
    # fmt: off
    priority_order = torch.tensor(
        [
            38, 35, 30, 25, 23, 31, 36, 37, 39,
            33, 28, 21, 20,  6, 15, 22, 29, 34,
            32, 18, 12, 10,  5,  9, 14, 19, 27,
            24, 13,  8,  2,  1,  3, 11, 17, 26,
            16,  7,  4,  0,  #
        ]
    )
    # fmt: on
    return priority_order


def _get_mask_size_ctx(n_spatial_ctx: int = 0) -> int:
    """Given the number of spatial contexts required, automatically compute
    the mask size around the coded value. The less contexts we need, the smaller
    the mask.
    """
    return MAX_ARM_MASK_SIZE


def _get_non_zero_pixel_ctx_index(n_spatial_ctx: int) -> Tensor:
    """Generate the relative index of the context pixel with respect to the
    actual pixel being decoded.

    1D tensor containing the indices of the non zero context. This corresponds to the one
    in the pattern above. This allows to use the index_select function, which is significantly
    faster than usual indexing.

    When we have n_spatial_ctx=N spatial context, we select only the pixels located at position
    [0, N-1] in the priority order map.

                Indices                                 Priority order

    0   1   2   3   4   5   6   7   8        38  35  30  25  23  31  36  37  39
    9   10  11  12  13  14  15  16  17       33  28  21  20   6  15  22  29  34
    18  19  20  21  22  23  24  25  26       32  18  12  10   5   9  14  19  27
    27  28  29  30  31  32  33  34  35       24  13   8   2   1   3  11  17  26
    36  37  38  39  *   x   x   x   x        16   7   4   0   *   x   x   x   x
    x   x   x   x   x   x   x   x   x         x   x   x   x   x   x   x   x   x
    x   x   x   x   x   x   x   x   x         x   x   x   x   x   x   x   x   x
    x   x   x   x   x   x   x   x   x         x   x   x   x   x   x   x   x   x
    x   x   x   x   x   x   x   x   x         x   x   x   x   x   x   x   x   x

    # # # Note: we automatically adjust the index of the selected neighbors to work with
    # # # the smallest possible arm_mask (computed through _get_mask_size_ctx(n_spatial_context))
    # # # in order to minimize the memory consumption.

    Args:
        n_spatial_ctx (int): Number of spatial context pixels

    Returns:
        Tensor: 1D tensor with the flattened index of the context pixels.
    """
    # center_pixel_idx = _get_center_pixel_ctx_index(n_spatial_ctx).item()
    center_pixel_idx = (_get_mask_size_ctx(n_spatial_ctx) ** 2 - 1) // 2
    possible_neighbors = torch.arange(center_pixel_idx)
    selected_neighbors = possible_neighbors[get_priority_order().argsort(stable=True)][
        :n_spatial_ctx
    ]

    return selected_neighbors
