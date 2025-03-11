# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch
from enc.io.format.yuv import DictTensorYUV
from torch import Tensor


@dataclass(kw_only=True)
class LossFunctionOutput():
    """Output for FrameEncoder.loss_function"""
    # ----- This is the important output
    # Optional to allow easy inheritance by FrameEncoderLogs
    loss: Optional[float] = None                                        # The RD cost to optimize

    # Any other data required to compute some logs, stored inside a dictionary
    mse: Optional[float] = None                         # Mean squared error                               [ / ]
    rate_latent_bpp: Optional[Dict[str, float]] = None  # Rate associated to the latent of each cool-chic  [bpp]
    total_rate_nn_bpp: float = 0.                       # Total rate associated to the all NNs of all cool-chic [bpp]

    # ==================== Not set by the init function ===================== #
    # Everything here is derived from the above metrics
    psnr_db: Optional[float] = field(init=False, default=None)                  # PSNR                            [ dB]
    total_rate_latent_bpp: Optional[float] = field(init=False, default=None)    # Overall rate of all the latents [bpp]
    total_rate_bpp: Optional[float] = field(init=False, default=None)           # Overall rate: latent & NNs      [bpp]
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        if self.mse is not None:
            self.psnr_db = -10.0 * math.log10(self.mse + 1e-10)

        if self.rate_latent_bpp is not None:
            self.total_rate_latent_bpp = sum(self.rate_latent_bpp.values())
        else:
            self.total_rate_latent_bpp = 0

        self.total_rate_bpp = self.total_rate_latent_bpp + self.total_rate_nn_bpp


def _compute_mse(
    x: Union[Tensor, DictTensorYUV], y: Union[Tensor, DictTensorYUV]
) -> Tensor:
    """Compute the Mean Squared Error between two images. Both images can
    either be a single tensor, or a dictionary of tensors with one for each
    color channel. In case of images with multiple channels, the final MSE
    is obtained by averaging the MSE for each color channel, weighted by the
    number of pixels. E.g. for YUV 420:
        MSE = (4 * MSE_Y + MSE_U + MSE_V) / 6

    Args:
        x (Union[Tensor, DictTensorYUV]): One of the two inputs
        y (Union[Tensor, DictTensorYUV]): The other input

    Returns:
        Tensor: One element tensor containing the MSE of x and y.
    """
    flag_420 = not (isinstance(x, Tensor))

    if not flag_420:
        return ((x - y) ** 2).mean()
    else:
        # Total number of pixels for all channels
        total_pixels_yuv = 0.0

        # MSE weighted by the number of pixels in each channels
        mse = torch.zeros((1), device=x.get("y").device)
        for (_, x_channel), (_, y_channel) in zip(x.items(), y.items()):
            n_pixels_channel = x_channel.numel()
            mse = (
                mse + torch.pow((x_channel - y_channel), 2.0).mean() * n_pixels_channel
            )
            total_pixels_yuv += n_pixels_channel
        mse = mse / total_pixels_yuv
        return mse


def loss_function(
    decoded_image: Union[Tensor, DictTensorYUV],
    rate_latent_bit: Dict[str, Tensor],
    target_image: Union[Tensor, DictTensorYUV],
    lmbda: float = 1e-3,
    total_rate_nn_bit: float = 0.,
    compute_logs: bool = False,
) -> LossFunctionOutput:
    """Compute the loss and a few other quantities. The loss equation is:

    .. math::

        \\mathcal{L} = ||\\mathbf{x} - \hat{\\mathbf{x}}||^2 + \\lambda
        (\\mathrm{R}(\hat{\\mathbf{x}}) + \\mathrm{R}_{NN}), \\text{ with }
        \\begin{cases}
            \\mathbf{x} & \\text{the original image}\\\\ \\hat{\\mathbf{x}} &
            \\text{the coded image}\\\\ \\mathrm{R}(\\hat{\\mathbf{x}}) &
            \\text{A measure of the rate of } \\hat{\\mathbf{x}} \\\\
                \\mathrm{R}_{NN} & \\text{The rate of the neural networks}
        \\end{cases}

    .. warning::

        There is no back-propagation through the term :math:`\\mathrm{R}_{NN}`.
        It is just here to be taken into account by the rate-distortion cost so
        that it better reflects the compression performance.

    Args:
        decoded_image: The decoded image, either as a Tensor for RGB or YUV444
            data, or as a dictionary of Tensors for YUV420 data.
        rate_latent_bit: Dictionary with the rate of each latent for each
            cool-chic decoder. Tensor with the rate of each latent value.
            The rate is in bit.
        target_image: The target image, either as a Tensor for RGB or YUV444
            data, or as a dictionary of Tensors for YUV420 data.
        lmbda: Rate constraint. Defaults to 1e-3.
        total_rate_nn_bit: Total rate of the NNs (arm + upsampling + synthesis)
            for all each cool-chic encoder. Rate is in bit. Defaults to 0.
        compute_logs: True to output a few more quantities beside the loss.
            Defaults to False.

    Returns:
        Object gathering the different quantities computed by this loss
        function. Chief among them: the loss itself.
    """
    if isinstance(target_image, Tensor):
        range_target = target_image.abs().max().item()
        if range_target > 1:
            target_min = target_image.min()
            target_max = target_image.max()

            decoded_image = (decoded_image - target_min) / (target_max - target_min)
            target_image = (target_image - target_min) / (target_max - target_min)

    mse = _compute_mse(decoded_image, target_image)

    if isinstance(decoded_image, Tensor):
        n_pixels = decoded_image.size()[-2] * decoded_image.size()[-1]
    else:
        n_pixels = decoded_image.get("y").size()[-2] * decoded_image.get("y").size()[-1]

    total_rate_latent_bit = torch.cat([v.sum().view(1) for _, v in rate_latent_bit.items()]).sum()
    rate_bpp = total_rate_latent_bit + total_rate_nn_bit
    rate_bpp = rate_bpp / n_pixels

    loss = mse + lmbda * rate_bpp

    # Construct the output module, only the loss is always returned
    rate_latent_bpp = None
    total_rate_nn_bpp = 0.
    if compute_logs:
        rate_latent_bpp = {
            k: v.detach().sum().item() / n_pixels
            for k, v in rate_latent_bit.items()
        }
        total_rate_nn_bpp = total_rate_nn_bit / n_pixels

    output = LossFunctionOutput(
        loss=loss,
        mse=mse.detach().item() if compute_logs else None,
        total_rate_nn_bpp=total_rate_nn_bpp,
        rate_latent_bpp=rate_latent_bpp,
    )

    return output
