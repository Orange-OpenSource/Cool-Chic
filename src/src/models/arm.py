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
from torch import Tensor, index_select, nn
from typing import List, Tuple
from einops import rearrange

from models.linear_layers import CustomLinear, CustomLinearResBlock
from models.quantizable_module import QuantizableModule
from utils.constants import ARMINT, POSSIBLE_Q_STEP_ARM_NN


class Arm(QuantizableModule):
    def __init__(self, input_ft: int, layers_dim: List[int], fpfm: int = 0):
        super().__init__(possible_q_steps=POSSIBLE_Q_STEP_ARM_NN)

        self.FPFM = fpfm # FIXED_POINT_FRACTIONAL_MULT # added for decode_network with torchscript.
        self.ARMINT = ARMINT
        self.qw = None
        self.qb = None

        # ======================== Construct the MLP ======================== #
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for out_ft in layers_dim:
            if input_ft == out_ft:
                layers_list.append(CustomLinearResBlock(input_ft, out_ft, self.FPFM))
            else:
                layers_list.append(CustomLinear(input_ft, out_ft, self.FPFM))
            layers_list.append(nn.ReLU())
            input_ft = out_ft

        # Construct the output layer. It always has 2 outputs (mu and scale)
        layers_list.append(CustomLinear(input_ft, 2, self.FPFM))
        self.mlp = nn.Sequential(*layers_list)
        # ======================== Construct the MLP ======================== #

    def set_quant(self, fpfm: int = 0):
        # Non-zero fpfm implies we're in fixed point mode. weights and biases are integers.
        # ARMINT False => the integers are stored in floats.
        self.FPFM = fpfm
        self.cnt = 0
        # Convert from float to fixed point int.
        for l in self.mlp.children():
            if isinstance(l, CustomLinearResBlock) or isinstance(l, CustomLinear) \
                or (hasattr(l, "original_name") and (l.original_name == "CustomLinearResBlock" or l.original_name == "CustomLinear")):
                l.scale = self.FPFM
                # shadow fixed point weights and biases.
                if self.ARMINT:
                    l.qw = torch.round(l.weight*l.scale).to(torch.int32)
                    l.qb = torch.round(l.bias*l.scale*l.scale).to(torch.int32)
                else:
                    l.qw = torch.round(l.weight*l.scale).to(torch.int32).to(torch.float)
                    l.qb = torch.round(l.bias*l.scale*l.scale).to(torch.int32).to(torch.float)

    def forward(self, x: Tensor) -> Tensor:
        # return self.mlp(x)
        if self.FPFM == 0:
            # Pure float processing.
            return self.mlp(x)

#        if False:
#            # Run in fp mode and clamp.
#            for l in self.mlp.children():
#                x = l(x)
#                x = torch.round(x*self.FPFM) / self.FPFM
#            return x

        # integer mode.
        xint = x.clone().detach()
        if self.ARMINT:
            xint = xint.to(torch.int32)*self.FPFM
        else:
            xint = (xint.to(torch.int32)*self.FPFM).to(torch.float)
        # print("ARMIN", self.cnt, x.shape, x)

        for l in self.mlp.children():
            xint = l(xint)
            #if self.cnt == 218:
            #    print("218 l out", xint)
        # xint = self.mlp(xint)

        # float the result.
        # print("ARMOUTI", self.cnt, xint.shape, xint)
        x = xint / self.FPFM
        # print("ARMOUTF", self.cnt, x.shape, x)
        self.cnt += 1
        return x

def get_neighbor(x: Tensor, mask_size: int, non_zero_pixel_ctx_idx: Tensor) -> Tensor:
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
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0.)
    # Shape of x_unfold is [B, C, H, W, mask_size, mask_size]
    x_unfold = x_pad.unfold(2, mask_size, step=1).unfold(3, mask_size, step=1)

    # Convert x_unfold to a 2D tensor: [Number of pixels, all neighbors]
    x_unfold = rearrange(
        x_unfold, 'b c h w mask_h mask_w -> (b c h w) (mask_h mask_w)'
    )

    # Select the pixels for which the mask is not zero
    # For a N x N mask, select only the first (N x N - 1) / 2 pixels
    # (those which aren't null)
    # ! index_select is way faster than normal indexing when using torch script
    neighbor = index_select(x_unfold, dim=1, index=non_zero_pixel_ctx_idx)
    return neighbor


def get_flat_latent_and_context(
    sent_latent: List[Tensor],
    mask_size: int,
    non_zero_pixel_ctx_idx: Tensor,
) -> Tuple[Tensor, Tensor]:
    """From a list of C tensors [1, 1, H_i, W_i], where H_i = H / 2 ** i,
    extract all the visible neighbors based on spatial_ctx_mask.

    Args:
        sent_latent (List[Tensor]): C tensors [1, 1, H_i, W_i], where H_i = H / 2 ** i
        mask_size (int): Virtual size of the kernel around the current coded latent.
            mask_size = 2 * n_ctx_rowcol - 1
        non_zero_pixel_ctx_idx (Tensor): [N] 1D tensor containing the indices
            of the non zero context pixels (i.e. floor(N ** 2 / 2) - 1).
            It looks like: [0, 1, ..., floor(N ** 2 / 2) - 1].
            This allows to use the index_select function, which is significantly
            faster than usual indexing.

    Returns:
        Tuple[Tensor, Tensor]:
            flat_latent [B], all the sent latent variables as a 1D tensor.
            flat_context [B, N_neighbors], the neighbors of each latent variable.
    """
    flat_latent_list: List[Tensor] = []
    flat_context_list: List[Tensor] = []

    # ============================= Context ============================= #
    # Get all the context as a single 2D vector of size [B, context size]
    for spatial_latent_i in sent_latent:
        flat_context_list.append(
            get_neighbor(spatial_latent_i, mask_size, non_zero_pixel_ctx_idx)
        )

        flat_latent_list.append(spatial_latent_i.view(-1))
    flat_context: Tensor = torch.cat(flat_context_list, dim=0)

    # Get all the B latent variables as a single one dimensional vector
    flat_latent: Tensor = torch.cat(flat_latent_list, dim=0)
    # ============================= Context ============================= #
    return flat_latent, flat_context


def laplace_cdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    """Compute the laplace cumulative evaluated in x. All parameters
    must have the same dimension.

    Args:
        x (Tensor): Where the cumulative if evaluated.
        loc (Tensor): Expectation.
        scale (Tensor): Scale

    Returns:
        Tensor: CDF(x, mu, scale)
    """
    return 0.5 - 0.5 * (x - loc).sign() * torch.expm1(-(x - loc).abs() / scale)

def get_mu_scale(raw_proba_param: Tensor) -> Tuple[Tensor, Tensor]:
    """From a raw tensor raw_proba_param [B, 2], split it into
    two halves (one for mu, one for scale) and reparameterize the scale
    so that it will always be positive.

    Args:
        raw_proba_param (Tensor): Output of the ARM [B, 2]

    Returns:
        mu: expectation of x [B]
        scale: scale of x [B]
    """
    mu, log_scale = [tmp.view(-1) for tmp in torch.split(raw_proba_param, 1, dim=1)]
    # no scale bigger than exp(-0.5 * -10) = 150
    # no scale smaller than exp(-0.5 * 13.81) = 1e-3
    scale = torch.exp(-0.5 * torch.clamp(log_scale, min=-10., max=13.8155))
    return mu, scale

def compute_rate(x: Tensor, raw_proba_param: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """From a raw tensor raw_proba_param [B, 2], split it into
    two halves (one for mu, one for scale) and reparameterize the scale
    so that it will always be positive.
    Then, estimate the entropy of x (in bits) when its distribution is estimated
    as Laplace(mu, scale) = MLP(dec_side_latent). x, mu and scale must have the same
    scale.

    Args:
        x (Tensor): Tensor whose entropy is measured [B]
        raw_proba_param (Tensor): Output of the ARM [B, 2]

    Returns:
        Tensor: rate (entropy) of x in bits. [B]
        mu: expectation of x [B]
        scale: scale of x [B]
    """
    mu, scale = get_mu_scale(raw_proba_param)
    proba = torch.clamp_min(
        laplace_cdf(x + 0.5, mu, scale) - laplace_cdf(x - 0.5, mu, scale),
        min=2 ** -16    # No value can cost more than 16 bits.
    )
    rate = -torch.log2(proba)
    return rate, mu, scale
