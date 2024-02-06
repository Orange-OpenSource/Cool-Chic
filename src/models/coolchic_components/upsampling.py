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
import numpy as np
from torch import Tensor, nn
from typing import List
from einops import rearrange

from models.coolchic_components.quantizable_module import QuantizableModule
from utils.misc import POSSIBLE_Q_STEP_UPS_NN

kernel_bilinear = [
    [0.0625, 0.1875, 0.1875, 0.0625],
    [0.1875, 0.5625, 0.5625, 0.1875],
    [0.1875, 0.5625, 0.5625, 0.1875],
    [0.0625, 0.1875, 0.1875, 0.0625],
]

kernel_bicubic = [
    [ 0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619],
    [ 0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857],
    [-0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498],
    [-0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479],
    [-0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479],
    [-0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498],
    [ 0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857],
    [ 0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619],
]

class Upsampling(QuantizableModule):
    def __init__(self, upsampling_kernel_size: int, static_upsampling_kernel: bool):
        """Create the upsampling layer. It can be learned or not, depending on
        the value of <static_upsampling_kernel>.

        In either case, the initialization of the upsampling kernel is based on
        <upsampling_kernel_size>. If upsampling_kernel_size >= 4 and < 8, we use
        the bilinear kernel with zero padding if necessary. Otherwise, if
        upsampling_kernel_size >= 8, we rely on the bicubic kernel.

        Args:
            upsampling_kernel_size (int): Upsampling kernel size. Should be >= 4
                and a multiple of two.
            static_upsampling_kernel (bool): If true, don't learn the upsampling kernel.
        """
        super().__init__(possible_q_steps=POSSIBLE_Q_STEP_UPS_NN)

        assert upsampling_kernel_size >= 4, f'Upsampling kernel size should be >= 4.' \
            f'Found {upsampling_kernel_size}'

        assert upsampling_kernel_size % 2 == 0, f'Upsampling kernel size should be even.' \
            f'Found {upsampling_kernel_size}'

        # adapted filter size
        self.static_upsampling_kernel = static_upsampling_kernel
        K = upsampling_kernel_size
        self.upsampling_padding = (K // 2, K // 2, K // 2, K // 2)
        self.upsampling_crop = (3 * K - 2) // 2


        # pad initial filter according to desired optimised kernel size
        if K < 8:
            K_init_filter= 4        # default bilinear upsampling filter
            kernek_init = kernel_bilinear
        else:
            K_init_filter = 8       # default bicubic upsampling filter
            kernek_init = kernel_bicubic

        tmpad=(K - K_init_filter) // 2
        kernel_pad=(tmpad, tmpad, tmpad, tmpad)
        upsampling_kernel=F.pad(torch.tensor(kernek_init), kernel_pad) # padded with zeros by default

        # compatible with transpose conv
        upsampling_kernel=torch.unsqueeze(upsampling_kernel, 0)
        upsampling_kernel=torch.unsqueeze(upsampling_kernel, 0)

        self.static_kernel = upsampling_kernel.clone()

        # Initialise layer and weights... even when static_upsampling_kernel is True
        # In this case, self.upsampling_layer will never be called (due to the if branch)
        # in the forward. Consequently, self.upsampling_layer.weight is completely set to
        # zero at the very end of the encoding, when quantizing the network and it won't cost any bit.
        self.upsampling_layer = nn.ConvTranspose2d(1,1,upsampling_kernel.shape ,bias=False,stride=2)
        self.upsampling_layer.weight.data = nn.Parameter(upsampling_kernel,requires_grad=True)


    def forward(self, decoder_side_latent: List[Tensor]) -> Tensor:
        """From a list of C [1, C', H_i, W_i] tensors, where H_i = H / 2 ** i abd
            W_i = W / 2 ** i, upsample each tensor to H * W. Then return the values
            as a 4d tensor = [1, C' x C, H_i, W_i]

        Args:
            decoder_side_latent (List[Tensor]): a list of C latent variables
                with resolution [1, C', H_i, W_i].

        Returns:
            Tensor: The [1, C' x C, H_i, W_i] synthesis input.
        """

        # The main idea is to invert the batch dimension (always equal to 1 in our case)
        # with the channel dimension (not always equal to 1) so that the same convolution
        # is applied independently on the batch dimension.
        upsampled_latent = rearrange(decoder_side_latent[-1], '1 c h w -> c 1 h w')

        for i in range(len(decoder_side_latent) - 1, 0, -1):
            # Our goal is to upsample <upsampled_latent> to the same resolution than <target_tensor>
            target_tensor = rearrange(decoder_side_latent[i - 1], '1 c h w -> c 1 h w')

            x_pad = F.pad(upsampled_latent, self.upsampling_padding, mode='replicate')

            # We don't use the learnable kernel, call the static kernel instead
            if self.static_upsampling_kernel:
                y_conv = F.conv_transpose2d(x_pad, self.static_kernel,stride=2)
            else:
                y_conv = self.upsampling_layer(x_pad) # the kernel is learned

            # crop to remove padding in convolution
            H, W = y_conv.size()[-2:]
            upsampled_latent = y_conv[
                :,
                :,
                self.upsampling_crop : H - self.upsampling_crop,
                self.upsampling_crop : W - self.upsampling_crop
            ]

            # crop to comply with higher resolution feature maps size before concatenation
            upsampled_latent = upsampled_latent[
                :,
                :,
                0 : target_tensor.shape[-2],
                0 : target_tensor.shape[-1]
            ]

            upsampled_latent = torch.cat((target_tensor, upsampled_latent), dim=0)

        return rearrange(upsampled_latent, 'c 1 h w -> 1 c h w')

