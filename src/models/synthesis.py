# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange
from typing import List
from models.linear_layers import CustomLinear, CustomLinearResBlock


class SynthesisMLP(torch.jit.ScriptModule):
    def __init__(self, input_ft: int, layers_dim: List[int]):
        """Instantiate a Synthesis MLP. It always has 3 (R, G, B) output features.

        Args:
            input_ft (int): Number of input dimensions. It corresponds to the number
                of latent grids.
            layers_dim (List[int]): List of the width of the hidden layers. Empty
                if no hidden layer (i.e. linear systems).
        """
        super().__init__()
        # ======================== Construct the MLP ======================== #
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for out_ft in layers_dim:
            if input_ft == out_ft:
                layers_list.append(CustomLinearResBlock(input_ft, out_ft))
            else:
                layers_list.append(CustomLinear(input_ft, out_ft))
            layers_list.append(nn.ReLU())
            input_ft = out_ft

        # Construct the output layer. It always has 3 outputs (RGB)
        layers_list.append(CustomLinear(input_ft, 3))
        self.mlp = nn.Sequential(*layers_list)
        # ======================== Construct the MLP ======================== #

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass for the Synthesis MLP.
        The input and output are 4D tensors. They are simply reshaped for the Linear
        layers.

        Args:
            x (Tensor): A [1, C, H, W] 4D tensor. With C the number of latent grids.

        Returns:
            Tensor: A [1, 3, H, W] tensor.
        """
        _, _, h, w = x.size()
        # Convert 4D to 2D for the MLP...
        x = rearrange(x, 'b c h w -> (b h w) c')
        x = self.mlp(x)
        # Go back from 2D to 4D. We output 3 features (i.e. RGB)
        x = rearrange(x, '(b h w) c -> b c h w', c = 3, h = h, w = w)
        return x


class STEQuantizer(torch.autograd.Function):
    """Actual quantization in the forward and set gradient to one ine the backward."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, training: bool):
        # training is a dummy parameters used to have the same signature for both
        # quantizer forward functions.
        ctx.save_for_backward(x)
        y = torch.round(x)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out, None  # No gradient with respect to <training> variable


class UniformNoiseQuantizer(torch.autograd.Function):
    """If training: use noise addition. Otherwise use actual quantization. Gradient is always one."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, training: bool):
        ctx.save_for_backward(x)
        if training:
            y = x + (torch.rand_like(x) - 0.5) if training else torch.round(x)
        else:
            y = torch.round(x)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out, None   # No gradient with respect to <training> variable


@torch.jit.script
def get_synthesis_input_latent(decoder_side_latent: List[Tensor]) -> Tensor:
    """From a list of C [1, 1, H_i, W_i] tensors, where H_i = H / 2 ** i abd
    W_i = W / 2 ** i, upsample each tensor to H * W. Then return the values
    as a 2d tensor [H * W, C]. This is the synthesis input

    Args:
        decoder_side_latent (List[Tensor]): a list of C latent variables
            with resolution [1, 1, H_i, W_i].

    Returns:
        Tensor: The [H * W, C] synthesis input.
    """
    # Start from the lowest resolution latent variable N, upsampled it to match
    # the size of the latent N - 1, concatenate them and do it again until all
    # latents have the same spatial dimension
    upsampled_latent: Tensor = decoder_side_latent[-1]
    for i in range(len(decoder_side_latent) - 1, 0, -1):
        target_tensor = decoder_side_latent[i - 1]
        upsampled_latent = F.interpolate(
            upsampled_latent,
            size=target_tensor.size()[-2:],
            mode='bicubic',
            align_corners=False
        )
        upsampled_latent = torch.cat((target_tensor, upsampled_latent), dim=1)

    return upsampled_latent
