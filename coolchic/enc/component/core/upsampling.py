# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from typing import List, OrderedDict

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from einops import rearrange
from torch import Tensor, nn


class _Parameterization_Symmetric_1d(nn.Module):
    """This module is not meant to be instantiated. It should rather be used
    through the ``torch.nn.utils.parametrize.register_parametrization()``
    function to reparameterize a N-element vector into a 2N-element (or 2N+1)
    symmetric vector. For instance:

        * x = a b c and target_k_size = 5 --> a b c b a
        * x = a b c and target_k_size = 6 --> a b c c b a

    Both these 5-element or 6-element vectors can be parameterize through
    a 3-element representation (a, b, c).
    """

    def __init__(self, target_k_size: int):
        """
        Args:
            target_k_size: Target size of the kernel after reparameterization.
        """

        super().__init__()
        self.target_k_size = target_k_size
        self.param_size = _Parameterization_Symmetric_1d.size_param_from_target(
            self.target_k_size
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return a longer, symmetric vector by concatenating x with a flipped
        version of itself.

        Args:
            x (Tensor): [N] tensor.

        Returns:
            Tensor: [2N] or [2N + 1] tensor, depending on self.target_k_size
        """

        # torch.fliplr requires to have a 2D kernel
        x_reversed = torch.fliplr(x.view(1, -1)).view(-1)

        kernel = torch.cat(
            [
                x,
                # a b c c b a if n is even or a b c b a if n is odd
                x_reversed[self.target_k_size % 2 :],
            ],
        )

        return kernel


    @classmethod
    def size_param_from_target(cls, target_k_size: int) -> int:
        """Return the size of the appropriate parameterization of a
        symmetric tensor with target_k_size elements. For instance:

            target_k_size = 6 ; parameterization size = 3 e.g. (a b c c b a)

            target_k_size = 7 ; parameterization size = 4 e.g. (a b c d c b a)

        Args:
            target_k_size (int): Size of the actual symmetric 1D kernel.

        Returns:
            int: Size of the underlying parameterization.
        """
        # For a kernel of size target_k_size = 2N, we need N values
        # e.g. 3 params a b c to parameterize a b c c b a.
        # For a kernel of size target_k_size = 2N + 1, we need N + 1 values
        # e.g. 4 params a b c d to parameterize a b c d c b a.
        return (target_k_size + 1) // 2



class UpsamplingSeparableSymmetricConv2d(nn.Module):
    """
    A conv2D which has a separable and symmetric *odd* kernel.

    Separable means that the 2D-kernel :math:`\mathbf{w}_{2D}` can be expressed
    as the outer product of a 1D kernel :math:`\mathbf{w}_{1D}`:

    .. math::

        \mathbf{w}_{2D} = \mathbf{w}_{1D} \otimes \mathbf{w}_{1D}.

    The 1D kernel :math:`\mathbf{w}_{1D}` is also symmetric. That is, the 1D
    kernel is something like :math:`\mathbf{w}_{1D} = \left(a\ b\ c\ b\ a\
    \\right).`

    The symmetric constraint is obtained through the module
    ``_Parameterization_Symmetric_1d``. The separable constraint is obtained by
    calling twice the 1D kernel.
    """
    def __init__(self, kernel_size: int):
        """
            kernel_size: Size of the kernel :math:`\mathbf{w}_{1D}` e.g. 7 to
                obtain a symmetrical, separable 7x7 filter. Must be odd!
        """
        super().__init__()

        assert (
            kernel_size % 2 == 1
        ), f"Upsampling kernel size must be odd, found {kernel_size}."

        self.target_k_size = kernel_size
        self.param_size = _Parameterization_Symmetric_1d.size_param_from_target(
            self.target_k_size
        )

        # -------- Instantiate empty parameters, set by the initialize function
        self.weight = nn.Parameter(
            torch.empty(self.param_size), requires_grad=True
        )

        self.bias = nn.Parameter(torch.empty(1), requires_grad=True)
        self.initialize_parameters()
        # -------- Instantiate empty parameters, set by the initialize function

        # Each time we call .weight, we'll call the forward of
        # _Parameterization_Symmetric_1d to get a symmetric kernel.
        parametrize.register_parametrization(
            self,
            "weight",
            _Parameterization_Symmetric_1d(target_k_size=self.target_k_size),
            # Unsafe because we change the data dimension, from N to 2N + 1
            unsafe=True,
        )

    def initialize_parameters(self) -> None:
        """
        Initialize the weights and the biases of the transposed convolution
        layer performing the upsampling.

            * Biases are always set to zero.

            * Weights are set to :math:`(0,\ 0,\ 0,\ \ldots, 1)` so that when the
              symmetric reparameterization is applied a Dirac kernel is obtained e.g.
              :math:`(0,\ 0,\ 0,\ \ldots, 1, \ldots, 0,\ 0,\ 0,)`.
        """
        # Zero everywhere except for the last coef
        w = torch.zeros_like(self.weight)
        w[-1] = 1
        self.weight = nn.Parameter(w, requires_grad=True)

        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Perform a "normal" 2D convolution, except that the underlying kernel
        is both separable & symmetrical. The actual implementation of the forward
        depends on ``self.training``.

        If we're training, we use a non-separable implementation. That is, we
        first compute the 2D kernel through an outer product and then use a
        single 2D convolution. This is more stable.

        If we're not training, we use two successive 1D convolutions.

        .. warning::

            There is a residual connexion in the forward.

        Args:
            x: [B, 1, H, W] tensor to be filtered. Must have one
                only channel.

        Returns:
            Tensor: Filtered tensor [B, 1, H, W].
        """
        k = self.weight.size()[0]
        weight = self.weight.view(1, -1)
        padding = k // 2

        # Train using non-separable (more stable)
        if self.training:
            # Kronecker product of (1 k) & (k 1) --> (k, k).
            # Then, two dummy dimensions are added to be compliant with conv2d
            # (k, k) --> (1, 1, k, k).
            kernel_2d = torch.kron(weight, weight.T).view((1, 1, k, k))

            # ! Note the residual connexion!
            return F.conv2d(x, kernel_2d, bias=None, stride=1, padding=padding) + x

        # Test through separable (less complex, for the flop counter)
        else:
            yw = F.conv2d(x, weight.view((1, 1, 1, k)), padding=(0, padding))

            # ! Note the residual connexion!
            return F.conv2d(yw, weight.view((1, 1, k, 1)), padding=(padding, 0)) + x


class UpsamplingSeparableSymmetricConvTranspose2d(nn.Module):
    """
    A TransposedConv2D which has a separable and symmetric *even* kernel.

    Separable means that the 2D-kernel :math:`\mathbf{w}_{2D}` can be expressed
    as the outer product of a 1D kernel :math:`\mathbf{w}_{1D}`:

    .. math::

        \mathbf{w}_{2D} = \mathbf{w}_{1D} \otimes \mathbf{w}_{1D}.

    The 1D kernel :math:`\mathbf{w}_{1D}` is also symmetric. That is, the 1D
    kernel is something like :math:`\mathbf{w}_{1D} = \left(a\ b\ c\ c\ b\ a\
    \\right).`

    The symmetric constraint is obtained through the module
    ``_Parameterization_Symmetric_1d``. The separable constraint is obtained by
    calling twice the 1D kernel.
    """

    def __init__(self, kernel_size: int):
        """
        Args:
            kernel_size: Upsampling kernel size. Shall be even and >= 4.
        """
        super().__init__()

        assert kernel_size >= 4 and not kernel_size % 2, (
            f"Upsampling kernel size shall be even and ≥4. Found {kernel_size}"
        )

        self.target_k_size = kernel_size
        self.param_size = _Parameterization_Symmetric_1d.size_param_from_target(
            self.target_k_size
        )

        # -------- Instantiate empty parameters, set by the initialize function
        self.weight = nn.Parameter(
            torch.empty(self.param_size), requires_grad=True
        )

        self.bias = nn.Parameter(torch.empty(1), requires_grad=True)
        self.initialize_parameters()
        # -------- Instantiate empty parameters, set by the initialize function

        # Each time we call .weight, we'll call the forward of
        # _Parameterization_Symmetric_1d to get a symmetric kernel.
        parametrize.register_parametrization(
            self,
            "weight",
            _Parameterization_Symmetric_1d(target_k_size=self.target_k_size),
            # Unsafe because we change the data dimension, from N to 2N + 1
            unsafe=True,
        )

    def initialize_parameters(self) -> None:
        """Initialize the parameters of a
        ``UpsamplingSeparableSymmetricConvTranspose2d`` layer.

            * Biases are always set to zero.

            * Weights are initialize as a (possibly padded) bilinear filter when
              ``target_k_size`` is 4 or 6, otherwise a bicubic filter is used.
        """
        # For a target kernel size of 4 or 6, we use a bilinear kernel as the
        # initialization. For bigger kernels, a bicubic kernel is used. In both
        # case we just initialize the left half of the kernel since these
        # filters are symmetrical
        if self.target_k_size < 8:
            kernel_core = torch.tensor([1.0 / 4.0, 3.0 / 4.0])
        else:
            kernel_core = torch.tensor([0.0351562, 0.1054687, -0.2617187, -0.8789063])

        # If target_k_size = 6, then param_size = 3 while kernel_core = 2
        # Thus we need to add zero_pad = 1 to the left of the kernel.
        zero_pad = self.param_size - kernel_core.size()[0]
        w = torch.zeros_like(self.weight)
        w[zero_pad:] = kernel_core
        self.weight = nn.Parameter(w, requires_grad=True)

        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the spatial upsampling (with scale 2) of an input with a
        single channel. Note that the upsampling filter is both symmetrical and
        separable. The actual implementation of the forward depends on
        ``self.training``.

        If we're training, we use a non-separable implementation. That is, we
        first compute the 2D kernel through an outer product and then use a
        single 2D convolution. This is more stable.

        If we're not training, we use two successive 1D convolutions.

        Args:
            x: Single channel input with shape :math:`(B, 1, H, W)`

        Returns:
            Upsampled version of the input with shape :math:`(B, 1, 2H, 2W)`
        """

        k = self.target_k_size  # kernel size
        P0 = k // 2  # could be 0 or k//2 as in legacy implementation
        C = 2 * P0 - 1 + k // 2  # crop side border k - 1 + k//2 (k=4, C=5  k=8, C=11)

        weight = self.weight.view(1, -1)

        if self.training:  # training using non-separable (more stable)
            kernel_2d = (torch.kron(weight, weight.T).view((1, 1, k, k)))

            x_pad = F.pad(x, (P0, P0, P0, P0), mode="replicate")
            yc = F.conv_transpose2d(x_pad, kernel_2d, stride=2)

            # crop to remove padding in convolution
            H, W = yc.size()[-2:]
            y = yc[
                :,
                :,
                C : H - C,
                C : W - C,
            ]

        else:  # testing through separable (less complex)
            # horizontal filtering
            x_pad = F.pad(x, (P0, P0, 0, 0), mode="replicate")
            yc = F.conv_transpose2d(x_pad, weight.view((1, 1, 1, k)), stride=(1, 2))
            W = yc.size()[-1]
            y = yc[
                :,
                :,
                :,
                C : W - C,
            ]

            # vertical filtering
            x_pad = F.pad(y, (0, 0, P0, P0), mode="replicate")
            yc = F.conv_transpose2d(x_pad, weight.view((1, 1, k, 1)), stride=(2, 1))
            H = yc.size()[-2]
            y = yc[:, :, C : H - C, :]

        return y


class Upsampling(nn.Module):
    """Create the upsampling module, its role is to upsampling the
    hierarchical latent variables :math:`\\hat{\\mathbf{y}} =
    \\{\\hat{\\mathbf{y}}_i \\in \\mathbb{Z}^{C_i \\times H_i \\times W_i},
    i = 0, \\ldots, L - 1\\}`, where :math:`L` is the number of latent
    resolutions and :math:`H_i = \\frac{H}{2^i}`, :math:`W_i =
    \\frac{W}{2^i}` with :math:`W, H` the width and height of the image.

    The Upsampling transforms this hierarchical latent variable
    :math:`\\hat{\\mathbf{y}}` into the dense representation
    :math:`\\hat{\\mathbf{z}}` as follows:

    .. math::

        \hat{\mathbf{z}} = f_{\\upsilon}(\hat{\mathbf{y}}), \\text{ with }
        \hat{\mathbf{z}} \\in \\mathbb{R}^{C \\times H \\times W} \\text {
        and } C = \\sum_i C_i.

    For a toy example with 3 latent grids (``--n_ft_per_res=1,1,1``), the
    overall diagram of the upsampling is as follows.

    .. code::

              +---------+
        y0 -> | TConv2d | -----+
              +---------+      |
                               v
              +--------+    +-----+    +---------+
        y1 -> | Conv2d | -> | cat | -> | TConv2d | -----+
              +--------+    +-----+    +---------+      |
                                                        v
                                         +--------+    +-----+    +---------+
        y2 ----------------------------> | Conv2d | -> | cat | -> | TConv2d | -> dense
                                         +--------+    +-----+    +---------+

    Where ``y0`` has the smallest resolution, ``y1`` has a resolution double of
    ``y0`` etc.

    There are two different sets of filters:

        * The TConvs filters actually perform the x2 upsampling. They are
          referred to as upsampling filters. Implemented using
          ``UpsamplingSeparableSymmetricConvTranspose2d``.

        * The Convs filters pre-process the signal prior to concatenation. They
          are referred to as pre-concatenation filters. Implemented using
          ``UpsamplingSeparableSymmetricConv2d``.

    Kernel sizes for the upsampling and pre-concatenation filters are modified
    through the ``--ups_k_size`` and ``--ups_preconcat_k_size`` arguments.

    Each upsampling filter and each pre-concatenation filter is different. They
    are all separable and symmetrical.

    Upsampling convolutions are initialized with a bilinear or bicubic kernel
    depending on the required requested ``ups_k_size``:

    * If ``ups_k_size >= 4 and ups_k_size < 8``, a
      bilinear kernel (with zero padding if necessary) is used an
      initialization.

    * If ``ups_k_size >= 8``, a bicubic kernel (with zero padding if
      necessary) is used an initialization.

    Pre-concatenation convolutions are initialized with a Dirac kernel.


    .. warning::

        * The ``ups_k_size`` must be at least 4 and a multiple of 2.

        * The ``ups_preconcat_k_size`` must be odd.
    """
    def __init__(
        self,
        ups_k_size: int,
        ups_preconcat_k_size: int,
        n_ups_kernel: int,
        n_ups_preconcat_kernel: int,
    ):
        """
        Args:
            ups_k_size: Upsampling (TransposedConv) kernel size. Should be
                even and >= 4.
            ups_preconcat_k_size: Pre-concatenation kernel size. Should be odd.
            n_ups_kernel: Number of different upsampling kernels. Usually it is
                set to the number of latent - 1 (because the full resolution
                latent is not upsampled). But this can also be set to one to
                share the same kernel across all variables.
            n_ups_preconcat_kernel: Number of different pre-concatenation
                filters. Usually it is set to the number of latent - 1 (because
                the smallest resolution is not filtered prior to concat).
                But this can also be set to one to share the same kernel across
                all variables.
        """
        super().__init__()

        # number of kernels for the lower and higher branches
        self.n_ups_kernel = n_ups_kernel
        self.n_ups_preconcat_kernel = n_ups_preconcat_kernel

        # Upsampling kernels = transpose conv2d
        self.conv_transpose2ds = nn.ModuleList(
            [
                UpsamplingSeparableSymmetricConvTranspose2d(ups_k_size)
                for _ in range(n_ups_kernel)
            ]
        )

        # Pre concatenation filters = conv2d
        self.conv2ds = nn.ModuleList(
            [
                UpsamplingSeparableSymmetricConv2d(ups_preconcat_k_size)
                for _ in range(self.n_ups_preconcat_kernel)
            ]
        )

    def forward(self, decoder_side_latent: List[Tensor]) -> Tensor:
        """Upsample a list of :math:`L` tensors, where the i-th
        tensor has a shape :math:`(B, C_i, \\frac{H}{2^i}, \\frac{W}{2^i})`
        to obtain a dense representation :math:`(B, \\sum_i C_i, H, W)`.
        This dense representation is ready to be used as the synthesis input.

        Args:
            decoder_side_latent: list of :math:`L` tensors with
                various shapes :math:`(B, C_i, \\frac{H}{2^i}, \\frac{W}{2^i})`

        Returns:
            Tensor: Dense representation :math:`(B, \\sum_i C_i, H, W)`.
        """
        # The main idea is to merge the channel dimension with the batch dimension
        # so that the same convolution is applied independently on the batch dimension.
        latent_reversed = list(reversed(decoder_side_latent))
        upsampled_latent = latent_reversed[0]  # start from smallest

        for idx, target_tensor in enumerate(latent_reversed[1:]):
            # Our goal is to upsample <upsampled_latent> to the same resolution than <target_tensor>
            x = rearrange(upsampled_latent, "b c h w -> (b c) 1 h w")
            x = self.conv_transpose2ds[idx % self.n_ups_kernel](x)

            x = rearrange(x, "(b c) 1 h w -> b c h w", b=upsampled_latent.shape[0])
            # Crop to comply with higher resolution feature maps size before concatenation
            x = x[:, :, : target_tensor.shape[-2], : target_tensor.shape[-1]]

            high_branch = self.conv2ds[idx % self.n_ups_preconcat_kernel](target_tensor)
            upsampled_latent = torch.cat((high_branch, x), dim=1)

        return upsampled_latent

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
        """Re-initialize **in place** the parameters of the upsampling."""
        for i in range(len(self.conv_transpose2ds)):
            self.conv_transpose2d[i].initialize_parameters()
        for i in range(len(self.conv2ds)):
            self.conv2ds[i].initialize_parameters()
