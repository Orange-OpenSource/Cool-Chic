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
from einops import rearrange
from torch import Tensor, nn


class UpsamplingConvTranspose2d(nn.Module):
    """Wrapper around the usual ``nn.TransposeConv2d`` layer. It performs a 2x
    upsampling of a latent variable with a **single** input and output channel.
    It can be learned or not, depending on the flag
    ``static_upsampling_kernel``. Its initialization depends on the requested
    kernel size. If the kernel size is 4 or 6, we use the bilinear kernel with
    zero padding if necessary. Otherwise, if the kernel size is 8 or bigger, we
    rely on the bicubic kernel.
    """

    kernel_bilinear = torch.tensor(
        [
            [0.0625, 0.1875, 0.1875, 0.0625],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.0625, 0.1875, 0.1875, 0.0625],
        ]
    )

    kernel_bicubic = torch.tensor(
        [
            [ 0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619],
            [ 0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857],
            [-0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498],
            [-0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479],
            [-0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479],
            [-0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498],
            [ 0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857],
            [ 0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619],
        ]
    )


    def __init__(
        self,
        upsampling_kernel_size: int,
        static_upsampling_kernel: bool
    ):
        """
        Args:
            upsampling_kernel_size: Upsampling kernel size. Should be >= 4
                and a multiple of two.
            static_upsampling_kernel: If true, don't learn the upsampling kernel.
        """
        super().__init__()

        assert upsampling_kernel_size >= 4, (
            f"Upsampling kernel size should be >= 4." f"Found {upsampling_kernel_size}"
        )

        assert upsampling_kernel_size % 2 == 0, (
            f"Upsampling kernel size should be even." f"Found {upsampling_kernel_size}"
        )

        self.upsampling_kernel_size = upsampling_kernel_size
        self.static_upsampling_kernel = static_upsampling_kernel

        # -------- Instantiate empty parameters, set by the initialize function
        self.weight = nn.Parameter(
            torch.empty(1, 1, upsampling_kernel_size, upsampling_kernel_size),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.empty((1)), requires_grad=True)
        self.initialize_parameters()
        # -------- Instantiate empty parameters, set by the initialize function

        # Keep initial weights if required by the self.static_upsampling kernel flag
        if self.static_upsampling_kernel:
            # register_buffer for automatic device management. We set persistent to false
            # to simply use the "automatically move to device" function, without
            # considering non_zero_pixel_ctx_index as a parameters (i.e. returned
            # by self.parameters())
            self.register_buffer("static_kernel", self.weight.data.clone(), persistent=False)
        else:
            self.static_kernel = None

    def initialize_parameters(self) -> None:
        """
        Initialize **in-place ** the weights and the biases of the transposed
        convolution layer performing the upsampling.

            - Biases are always set to zero.

            - Weights are set to a (padded) bicubic kernel if kernel size is at
              least 8. If kernel size is greater than or equal to 4, weights are
              set to a (padded) bilinear kernel.
        """
        # -------- bias is always set to zero (and in fact never ever used)
        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)

        # -------- Weights are initialized to bicubic or bilinear
        # adapted filter size
        K = self.upsampling_kernel_size
        self.upsampling_padding = (K // 2, K // 2, K // 2, K // 2)
        self.upsampling_crop = (3 * K - 2) // 2

        if K < 8:
            kernel_init = UpsamplingConvTranspose2d.kernel_bilinear
        else:
            kernel_init = UpsamplingConvTranspose2d.kernel_bicubic

        # pad initial filter according to desired kernel size
        tmpad = (K - kernel_init.size()[0]) // 2
        upsampling_kernel = F.pad(
            kernel_init.clone().detach(),
            (tmpad, tmpad, tmpad, tmpad),
            mode="constant",
            value=0.0,
        )

        # 4D kernel to be compatible with transpose convolution
        upsampling_kernel = rearrange(upsampling_kernel, "k_h k_w -> 1 1 k_h k_w")
        self.weight = nn.Parameter(upsampling_kernel, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the spatial upsampling (with scale 2) of an input with a
        single channel.

        Args:
            x: Single channel input with shape :math:`(B, 1, H, W)`

        Returns:
            Upsampled version of the input with shape :math:`(B, 1, 2H, 2W)`
        """
        upsampling_weight = (
            self.static_kernel if self.static_upsampling_kernel else self.weight
        )

        x_pad = F.pad(x, self.upsampling_padding, mode="replicate")
        y_conv = F.conv_transpose2d(x_pad, upsampling_weight, stride=2)

        # crop to remove padding in convolution
        H, W = y_conv.size()[-2:]
        results = y_conv[
            :,
            :,
            self.upsampling_crop : H - self.upsampling_crop,
            self.upsampling_crop : W - self.upsampling_crop,
        ]

        return results


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

    The upsampling relies on a single custom transpose convolution
    ``UpsamplingConvTranspose2d`` performing a 2x upsampling of a 1-channel
    input. This transpose convolution is called over and over to upsampling
    each channel of each resolution until they reach the required :math:`H
    \\times W` dimensions.

    The kernel of the ``UpsamplingConvTranspose2d`` depending on the value
    of the flag ``static_upsampling_kernel``. In either case, the kernel
    initialization is based on well-known bilinear or bicubic kernel
    depending on the requested ``upsampling_kernel_size``:

    * If ``upsampling_kernel_size >= 4 and upsampling_kernel_size < 8``, a
      bilinear kernel (with zero padding if necessary) is used an
      initialization.

    * If ``upsampling_kernel_size >= 8``, a bicubic kernel (with zero padding if
      necessary) is used an initialization.

    .. warning::

        The ``upsampling_kernel_size`` must be at least 4 and a multiple of 2.
    """


    def __init__(self, upsampling_kernel_size: int, static_upsampling_kernel: bool):
        """
        Args:
            upsampling_kernel_size: Upsampling kernel size. Should be bigger or
                equal to 4 and a multiple of two.
            static_upsampling_kernel: If true, don't learn the upsampling
                kernel.
        """
        super().__init__()

        self.conv_transpose2d = UpsamplingConvTranspose2d(
            upsampling_kernel_size, static_upsampling_kernel
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
        for target_tensor in latent_reversed[1:]:
            # Our goal is to upsample <upsampled_latent> to the same resolution than <target_tensor>
            x = rearrange(upsampled_latent, "b c h w -> (b c) 1 h w")
            x = self.conv_transpose2d(x)
            x = rearrange(x, "(b c) 1 h w -> b c h w", b=upsampled_latent.shape[0])
            # Crop to comply with higher resolution feature maps size before concatenation
            x = x[:, :, : target_tensor.shape[-2], : target_tensor.shape[-1]]
            upsampled_latent = torch.cat((target_tensor, x), dim=1)
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
        self.conv_transpose2d.initialize_parameters()
