# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from typing import OrderedDict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, index_select, nn


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
            residual: True to add a residual connexion to the layer. Defaults to
                False.
        """

        super().__init__()

        self.residual = residual

        # -------- Instantiate empty parameters, set by the initialize function
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels), requires_grad=True
        )
        self.bias = nn.Parameter(torch.empty((out_channels)), requires_grad=True)
        self.initialize_parameters()
        # -------- Instantiate empty parameters, set by the initialize function

    def initialize_parameters(self) -> None:
        """Initialize **in place** the weight and the bias of the linear layer.

        * Biases are always set to zero.

        * Weights are set to zero if ``residual == True``. Otherwise, sample
          from the Normal distribution: :math:`\\mathbf{W} \sim \\mathcal{N}(0,
          \\tfrac{1}{(C_{out})^4})`.
        """
        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)
        if self.residual:
            self.weight = nn.Parameter(
                torch.zeros_like(self.weight), requires_grad=True
            )
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

class ArmIntLinear(nn.Module):
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
        fpfm: int = 0,
        pure_int: bool = False,
        residual: bool = False,
    ):
        """
        Args:
            in_channels: Number of input features :math:`C_{in}`.
            out_channels: Number of output features :math:`C_{out}`.
            fpfm: Internal stuff for integer computation.  **No need to modify
                this**. Defaults to 0.
            residual: True to add a residual connexion to the layer. Defaults to
                False.
        """

        super().__init__()

        self.fpfm = fpfm
        self.pure_int = pure_int
        self.residual = residual

        # -------- Instantiate empty parameters, set by a later load
        if self.pure_int:
            self.weight = nn.Parameter(
                torch.empty((out_channels, in_channels), dtype=torch.int32), requires_grad=False
            )
            self.bias = nn.Parameter(torch.empty((out_channels), dtype=torch.int32), requires_grad=False)
        else:
            self.weight = nn.Parameter(
                torch.empty((out_channels, in_channels), dtype=torch.float), requires_grad=False
            )
            self.bias = nn.Parameter(torch.empty((out_channels), dtype=torch.float), requires_grad=False)
        # -------- Instantiate empty parameters, set by a later load


    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of this layer.

        Args:
            x: Input tensor of shape :math:`[B, C_{in}]`.

        Returns:
            Tensor with shape :math:`[B, C_{out}]`.
        """
        if self.residual:
            xx = F.linear(x, self.weight, bias=self.bias) + x*self.fpfm
        else:
            xx = F.linear(x, self.weight, bias=self.bias)

        # Renorm by fpfm after our (x*fpfm)*(qw*fpfm) multiplication.
        # WE MAKE INTEGER DIVISION OBEY C++ (TO-ZERO) SEMANTICS, NOT PYTHON (TO-NEGATIVE-INFINITY) SEMANTICS
        if self.pure_int:
            xx = xx + torch.sign(xx)*self.fpfm//2
            # We separate out -ve and non-ve.
            neg_result = -((-xx)//self.fpfm)
            pos_result = xx//self.fpfm
            result = torch.where(xx < 0, neg_result, pos_result)
        else:
            xx = xx + torch.sign(xx)*self.fpfm/2
            # We separate out -ve and non-ve.
            neg_result = -((-xx)/self.fpfm)
            pos_result = xx/self.fpfm
            result = torch.where(xx < 0, neg_result, pos_result)
            result = result.to(torch.int32).to(torch.float)

        return result

class Arm(nn.Module):
    """Instantiate an autoregressive probability module, modelling the
    conditional distribution :math:`p_{\\psi}(\\hat{y}_i \\mid
    \\mathbf{c}_i)` of a (quantized) latent pixel :math:`\\hat{y}_i`,
    conditioned on neighboring already decoded context pixels
    :math:`\\mathbf{c}_i \in \\mathbb{Z}^C`, where :math:`C` denotes the
    number of context pixels.

    The distribution :math:`p_{\\psi}` is assumed to follow a Laplace
    distribution, parameterized by an expectation :math:`\\mu` and a scale
    :math:`b`, where the scale and the variance :math:`\\sigma^2` are
    related as follows :math:`\\sigma^2 = 2 b ^2`.

    The parameters of the Laplace distribution for a given latent pixel
    :math:`\\hat{y}_i` are obtained by passing its context pixels
    :math:`\\mathbf{c}_i` through an MLP :math:`f_{\\psi}`:

    .. math::

        p_{\\psi}(\\hat{y}_i \\mid \\mathbf{c}_i) \sim \mathcal{L}(\\mu_i,
        b_i), \\text{ where } \\mu_i, b_i = f_{\\psi}(\\mathbf{c}_i).

    .. attention::

        The MLP :math:`f_{\\psi}` has a few constraint on its architecture:

        * The width of all hidden layers (i.e. the output of all layers except
          the final one) are identical to the number of pixel contexts
          :math:`C`;

        * All layers except the last one are residual layers, followed by a
          ``ReLU`` non-linearity;

        * :math:`C` must be at a multiple of 8.

    The MLP :math:`f_{\\psi}` is made of custom Linear layers instantiated
    from the ``ArmLinear`` class.
    """


    def __init__(self, dim_arm: int, n_hidden_layers_arm: int):
        """
        Args:
            dim_arm: Number of context pixels AND dimension of all hidden
                layers :math:`C`.
            n_hidden_layers_arm: Number of hidden layers. Set it to 0 for
                a linear ARM.
        """
        super().__init__()

        assert dim_arm % 8 == 0, (
            f"ARM context size and hidden layer dimension must be "
            f"a multiple of 8. Found {dim_arm}."
        )

        # ======================== Construct the MLP ======================== #
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for i in range(n_hidden_layers_arm):
            layers_list.append(ArmLinear(dim_arm, dim_arm, residual=True))
            layers_list.append(nn.ReLU())

        # Construct the output layer. It always has 2 outputs (mu and scale)
        layers_list.append(ArmLinear(dim_arm, 2, residual=False))
        self.mlp = nn.Sequential(*layers_list)
        # ======================== Construct the MLP ======================== #

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform the auto-regressive module (ARM) forward pass. The ARM takes
        as input a tensor of shape :math:`[B, C]` i.e. :math:`B` contexts with
        :math:`C` context pixels. ARM outputs :math:`[B, 2]` values correspond
        to :math:`\\mu, b` for each of the :math:`B` input pixels.

        .. warning::

            Note that the ARM expects input to be flattened i.e. spatial
            dimensions :math:`H, W` are collapsed into a single batch-like
            dimension :math:`B = HW`, leading to an input of shape
            :math:`[B, C]`, gathering the :math:`C` contexts for each of the
            :math:`B` pixels to model.

        .. note::

            The ARM MLP does not output directly the scale :math:`b`. Denoting
            :math:`s` the raw output of the MLP, the scale is obtained as
            follows:

            .. math::

                b = e^{x - 4}

        Args:
            x: Concatenation of all input contexts
                :math:`\\mathbf{c}_i`. Tensor of shape :math:`[B, C]`.

        Returns:
            Concatenation of all Laplace distributions param :math:`\\mu, b`.
            Tensor of shape :math:([B]). Also return the *log scale*
            :math:`s` as described above. Tensor of shape :math:`(B)`
        """
        raw_proba_param = self.mlp(x)
        mu = raw_proba_param[:, 0]
        log_scale = raw_proba_param[:, 1]

        # no scale smaller than exp(-4.6) = 1e-2 or bigger than exp(5.01) = 150
        scale = torch.exp(torch.clamp(log_scale - 4, min=-4.6, max=5.0))

        return mu, scale, log_scale

    def get_param(self) -> OrderedDict[str, Tensor]:
        """Return **a copy** of the weights and biases inside the module.

        Returns:
            A copy of all weights & biases in the layers.
        """
        # Detach & clone to create a copy
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        """Replace the current parameters of the module with param.

        Args:
            param: Parameters to be set.
        """
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
        """Re-initialize in place the parameters of all the ArmLinear layer."""
        for layer in self.mlp.children():
            if isinstance(layer, ArmLinear):
                layer.initialize_parameters()

class ArmInt(nn.Module):
    """Instantiate an autoregressive probability module, modelling the
    conditional distribution :math:`p_{\\psi}(\\hat{y}_i \\mid
    \\mathbf{c}_i)` of a (quantized) latent pixel :math:`\\hat{y}_i`,
    conditioned on neighboring already decoded context pixels
    :math:`\\mathbf{c}_i \in \\mathbb{Z}^C`, where :math:`C` denotes the
    number of context pixels.

    The distribution :math:`p_{\\psi}` is assumed to follow a Laplace
    distribution, parameterized by an expectation :math:`\\mu` and a scale
    :math:`b`, where the scale and the variance :math:`\\sigma^2` are
    related as follows :math:`\\sigma^2 = 2 b ^2`.

    The parameters of the Laplace distribution for a given latent pixel
    :math:`\\hat{y}_i` are obtained by passing its context pixels
    :math:`\\mathbf{c}_i` through an MLP :math:`f_{\\psi}`:

    .. math::

        p_{\\psi}(\\hat{y}_i \\mid \\mathbf{c}_i) \sim \mathcal{L}(\\mu_i,
        b_i), \\text{ where } \\mu_i, b_i = f_{\\psi}(\\mathbf{c}_i).

    .. attention::

        The MLP :math:`f_{\\psi}` has a few constraint on its architecture:

        * The width of all hidden layers (i.e. the output of all layers except
          the final one) are identical to the number of pixel contexts
          :math:`C`;

        * All layers except the last one are residual layers, followed by a
          ``ReLU`` non-linearity;

        * :math:`C` must be at a multiple of 8.

    The MLP :math:`f_{\\psi}` is made of custom Linear layers instantiated
    from the ``ArmLinear`` class.
    """

    def __init__(self, dim_arm: int, n_hidden_layers_arm: int, fpfm: int, pure_int: bool):
        """
        Args:
            dim_arm: Number of context pixels AND dimension of all hidden
                layers :math:`C`.
            n_hidden_layers_arm: Number of hidden layers. Set it to 0 for
                a linear ARM.
        """
        super().__init__()

        assert dim_arm % 8 == 0, (
            f"ARM context size and hidden layer dimension must be "
            f"a multiple of 8. Found {dim_arm}."
        )

        self.FPFM = fpfm # fixed-point: multiplication to get int.
        self.pure_int = pure_int # weights and biases are actual int (cpu only), or just int values in floats (gpu friendly).

        # ======================== Construct the MLP ======================== #
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for i in range(n_hidden_layers_arm):
            layers_list.append(ArmIntLinear(dim_arm, dim_arm, self.FPFM, self.pure_int, residual=True))
            layers_list.append(nn.ReLU())

        # Construct the output layer. It always has 2 outputs (mu and scale)
        layers_list.append(ArmIntLinear(dim_arm, 2, self.FPFM, self.pure_int, residual=False))
        self.mlp = nn.Sequential(*layers_list)
        # ======================== Construct the MLP ======================== #

    def set_param_from_float(self, float_param: OrderedDict[str, Tensor]) -> None:
        # We take floating point values here, and convert them to ints.

        # floating point params.  We convert to fixed-point integer and store them.
        integerised_param = {}
        for k in float_param:
            if "weight" in k:
                float_v = float_param[k]*self.FPFM
            else:
                float_v = float_param[k]*self.FPFM*self.FPFM

            float_v = float_v + torch.sign(float_v)*0.5
            neg_result = -(-float_v).to(torch.int32)
            pos_result = float_v.to(torch.int32)
            int_v = torch.where(float_v < 0, neg_result, pos_result)
            if not self.pure_int:
                int_v = int_v.to(torch.float)
            integerised_param[k] = nn.parameter.Parameter(int_v, requires_grad=False)

        self.load_state_dict(integerised_param, assign=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform the auto-regressive module (ARM) forward pass. The ARM takes
        as input a tensor of shape :math:`[B, C]` i.e. :math:`B` contexts with
        :math:`C` context pixels. ARM outputs :math:`[B, 2]` values correspond
        to :math:`\\mu, b` for each of the :math:`B` input pixels.

        .. warning::

            Note that the ARM expects input to be flattened i.e. spatial
            dimensions :math:`H, W` are collapsed into a single batch-like
            dimension :math:`B = HW`, leading to an input of shape
            :math:`[B, C]`, gathering the :math:`C` contexts for each of the
            :math:`B` pixels to model.

        .. note::

            The ARM MLP does not output directly the scale :math:`b`. Denoting
            :math:`s` the raw output of the MLP, the scale is obtained as
            follows:

            .. math::

                b = e^{x - 4}

        Args:
            x: Concatenation of all input contexts
                :math:`\\mathbf{c}_i`. Tensor of shape :math:`[B, C]`.

        Returns:
            Concatenation of all Laplace distributions param :math:`\\mu, b`.
            Tensor of shape :math:([B]). Also return the *log scale*
            :math:`s` as described above. Tensor of shape :math:`(B)`
        """
        xint = x.clone().detach()
        xint = xint*self.FPFM
        if self.pure_int:
            xint = xint.to(torch.int32)

        for idx_l, layer in enumerate(self.mlp.children()):
            xint = layer(xint)

        # float the result.
        raw_proba_param = xint / self.FPFM

        mu = raw_proba_param[:, 0]
        log_scale = raw_proba_param[:, 1]

        # no scale smaller than exp(-4.6) = 1e-2 or bigger than exp(5.01) = 150
        scale = torch.exp(torch.clamp(log_scale - 4, min=-4.6, max=5.0))

        return mu, scale, log_scale

    def get_param(self) -> OrderedDict[str, Tensor]:
        """Return **a copy** of the weights and biases inside the module.

        Returns:
            A copy of all weights & biases in the layers.
        """
        # Detach & clone to create a copy
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        """Replace the current parameters of the module with param.

        Args:
            param: Parameters to be set.
        """
        self.load_state_dict(param)

@torch.jit.script
def _get_neighbor(x: Tensor, mask_size: int, non_zero_pixel_ctx_idx: Tensor) -> Tensor:
    """Use the unfold function to extract the neighbors of each pixel in x.

    Args:
        x (Tensor): [1, 1, H, W] feature map from which we wish to extract the
            neighbors
        mask_size (int): Virtual size of the kernel around the current coded latent.
            mask_size = 2 * n_ctx_rowcol - 1
        non_zero_pixel_ctx_idx (Tensor): [N] 1D tensor containing the indices
            of the non zero context pixels (i.e. floor(N ** 2 / 2) - 1).
            It looks like: [0, 1, ..., floor(N ** 2 / 2) - 1].
            This allows to use the index_select function, which is significantly
            faster than usual indexing.

    Returns:
        torch.tensor: [H * W, floor(N ** 2 / 2) - 1] the spatial neighbors
            the floor(N ** 2 / 2) - 1 neighbors of each H * W pixels.
    """
    pad = int((mask_size - 1) / 2)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)

    # Shape of x_unfold is [B, C, H, W, mask_size, mask_size] --> [B * C * H * W, mask_size * mask_size]
    # reshape is faster than einops.rearrange
    x_unfold = (
        x_pad.unfold(2, mask_size, step=1)
        .unfold(3, mask_size, step=1)
        .reshape(-1, mask_size * mask_size)
    )

    # Convert x_unfold to a 2D tensor: [Number of pixels, all neighbors]
    # This is slower than reshape above
    # x_unfold = rearrange(
    #     x_unfold,
    #     'b c h w mask_h mask_w -> (b c h w) (mask_h mask_w)'
    # )

    # Select the pixels for which the mask is not zero
    # For a N x N mask, select only the first (N x N - 1) / 2 pixels
    # (those which aren't null)
    neighbor = index_select(x_unfold, dim=1, index=non_zero_pixel_ctx_idx)
    return neighbor


@torch.jit.script
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


def _get_non_zero_pixel_ctx_index(dim_arm: int) -> Tensor:
    """Generate the relative index of the context pixel with respect to the
    actual pixel being decoded.

    1D tensor containing the indices of the non zero context. This corresponds to the one
    in the pattern above. This allows to use the index_select function, which is significantly
    faster than usual indexing.

    0   1   2   3   4   5   6   7   8
    9   10  11  12  13  14  15  16  17
    18  19  20  21  22  23  24  25  26
    27  28  29  30  31  32  33  34  35
    36  37  38  39  *   x   x   x   x
    x   x   x   x   x   x   x   x   x
    x   x   x   x   x   x   x   x   x
    x   x   x   x   x   x   x   x   x
    x   x   x   x   x   x   x   x   x


    Args:
        dim_arm (int): Number of context pixels

    Returns:
        Tensor: 1D tensor with the flattened index of the context pixels.
    """

    if dim_arm == 8:
        return torch.tensor(
            [            13,
                         22,
                     30, 31, 32,
             37, 38, 39, #
            ]
        )

    elif dim_arm == 16:
        return torch.tensor(
            [
                            13, 14,
                    20, 21, 22, 23, 24,
                28, 29, 30, 31, 32, 33,
                37, 38, 39, #
            ]
        )

    elif dim_arm == 24:
        return torch.tensor(
            [
                                4 ,
                        11, 12, 13, 14, 15,
                    19, 20, 21, 22, 23, 24, 25,
                    28, 29, 30, 31, 32, 33, 34,
                36, 37, 38, 39, #
            ]
        )

    elif dim_arm == 32:
        return torch.tensor(
            [
                        2 , 3 , 4 , 5 ,
                    10, 11, 12, 13, 14, 15, 16,
                    19, 20, 21, 22, 23, 24, 25, 26,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, #
            ]
        )
