# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from enc.utils.codingstructure import DictTensorYUV
from torch import Tensor


@dataclass(kw_only=True)
class LossFunctionOutput():
    """Output for FrameEncoder.loss_function"""
    # ----- This is the important output
    # Optional to allow easy inheritance by FrameEncoderLogs
    loss: Optional[float] = None                                        # The RD cost to optimize

    # Any other data required to compute some logs, stored inside a dictionary
    mse: Optional[float] = None                                         # Mean squared error                     [ / ]
    rate_nn_bpp: Optional[float] = None                                 # Rate associated to the neural networks [bpp]
    rate_latent_bpp: Optional[float] = None                             # Rate associated to the latent          [bpp]

    # ==================== Not set by the init function ===================== #
    # Everything here is derived from the above metrics
    psnr_db: Optional[float] = field(init=False, default=None)          # PSNR                                  [ dB]
    total_rate_bpp: Optional[float] = field(init=False, default=None)   # Overall rate: latent & NNs            [bpp]
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        if self.mse is not None:
            self.psnr_db = -10.0 * math.log10(self.mse)

        if self.rate_nn_bpp is not None and self.rate_latent_bpp is not None:
            self.total_rate_bpp = self.rate_nn_bpp + self.rate_latent_bpp


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
    rate_latent_bit: Tensor,
    target_image: Union[Tensor, DictTensorYUV],
    lmbda: float = 1e-3,
    rate_mlp_bit: float = 0.0,
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
        rate_latent_bit: Tensor with the rate of each latent value.
            The rate is in bit.
        target_image: The target image, either as a Tensor for RGB or YUV444
            data, or as a dictionary of Tensors for YUV420 data.
        lmbda: Rate constraint. Defaults to 1e-3.
        rate_mlp_bit: Sum of the rate allocated for the different neural
            networks. Rate is in bit. Defaults to 0.0.
        compute_logs: True to output a few more quantities beside the loss.
            Defaults to False.

    Returns:
        Object gathering the different quantities computed by this loss
        function. Chief among them: the loss itself.
    """

    mse = _compute_mse(decoded_image, target_image)

    if isinstance(decoded_image, Tensor):
        n_pixels = decoded_image.size()[-2] * decoded_image.size()[-1]
    else:
        n_pixels = decoded_image.get("y").size()[-2] * decoded_image.get("y").size()[-1]

    rate_bpp = (rate_latent_bit.sum() + rate_mlp_bit) / n_pixels

    loss = mse + lmbda * rate_bpp

    # Construct the output module, only the loss is always returned
    output = LossFunctionOutput(
        loss=loss,
        mse=mse.detach().item() if compute_logs else None,
        rate_nn_bpp=rate_mlp_bit / n_pixels if compute_logs else None,
        rate_latent_bpp=rate_latent_bit.detach().sum().item() / n_pixels
        if compute_logs
        else None,
    )

    return output
