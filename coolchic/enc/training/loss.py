# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import torch
from enc.training.metrics.msssim import ms_ssim_fn
from enc.utils.codingstructure import DictTensorYUV
from torch import Tensor


TUNING_MODES = Literal["mse", "mse_msssim"]

@dataclass(kw_only=True)
class LossFunctionOutput():
    """Output for FrameEncoder.loss_function"""
    # ----- This is the important output
    # Optional to allow easy inheritance by FrameEncoderLogs
    loss: Optional[float] = None                                        # The RD cost to optimize

    # Any other data required to compute some logs, stored inside a dictionary
    mse: Optional[float] = None                                         # Mean squared error                     [ / ]
    ms_ssim: Optional[float] = None                                     # Multi-scale Structural Similarity Metric [ / ]
    rate_nn_bpp: Optional[float] = None                                 # Rate associated to the neural networks [bpp]
    rate_latent_bpp: Optional[float] = None                             # Rate associated to the latent          [bpp]

    # ==================== Not set by the init function ===================== #
    # Everything here is derived from the above metrics
    psnr_db: Optional[float] = field(init=False, default=None)          # PSNR                                  [ dB]
    ms_ssim_db: Optional[float] = field(init=False, default=None)       # MS-SSIM but with a log scale          [ dB]
    total_rate_bpp: Optional[float] = field(init=False, default=None)   # Overall rate: latent & NNs            [bpp]
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        if self.mse is not None:
            self.psnr_db = -10.0 * math.log10(self.mse)

        if self.ms_ssim is not None:
            self.ms_ssim_db = -10.0 * math.log10(1 - self.ms_ssim)

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


def _compute_ms_ssim(
    x: Union[Tensor, DictTensorYUV], y: Union[Tensor, DictTensorYUV]
) -> Tensor:
    """Compute the MS-SSIM between two images. Both images can
    either be a single tensor, or a dictionary of tensors with one for each
    color channel. In case of images with multiple channels, the final MS-SSIM
    is obtained by averaging the MS-SSIM  for each color channel, weighted by the
    number of pixels. E.g. for YUV 420:
        MS-SSIM  = (4 * MS-SSIM_Y + MS-SSIM_U + MS-SSIM_V) / 6

    Args:
        x (Union[Tensor, DictTensorYUV]): One of the two inputs
        y (Union[Tensor, DictTensorYUV]): The other input

    Returns:
        Tensor: One element tensor containing the MS-SSIM of x and y.
    """
    flag_420 = not (isinstance(x, Tensor))

    if not flag_420:
        return ms_ssim_fn(x, y, data_range=1.0)
    else:
        # Total number of pixels for all channels
        total_pixels_yuv = 0.0

        # MS-SSIM weighted by the number of pixels in each channels
        ms_ssim = torch.zeros((1), device=x.get("y").device)
        for (_, x_channel), (_, y_channel) in zip(x.items(), y.items()):
            n_pixels_channel = x_channel.numel()
            ms_ssim = (
                ms_ssim + ms_ssim_fn(x_channel, y_channel, data_range=1.0) * n_pixels_channel
            )
            total_pixels_yuv += n_pixels_channel
        ms_ssim = ms_ssim / total_pixels_yuv
        return ms_ssim


def loss_function(
    decoded_image: Union[Tensor, DictTensorYUV],
    rate_latent_bit: Tensor,
    target_image: Union[Tensor, DictTensorYUV],
    lmbda: float = 1e-3,
    rate_mlp_bit: float = 0.0,
    compute_logs: bool = False,
    tune: TUNING_MODES = "mse",
) -> LossFunctionOutput:
    """Compute the loss and a few other quantities. The loss equation is:

    .. math::

        \\mathcal{L} = \\mathrm{D}(\hat{\\mathbf{x}}, \\mathbf{x}) + \\lambda
        (\\mathrm{R}(\hat{\\mathbf{x}}) + \\mathrm{R}_{NN}), \\text{ with }
        \\begin{cases}
            \\mathbf{x} & \\text{the original image}\\\\ \\hat{\\mathbf{x}} &
            \\text{the coded image}\\\\ \\mathrm{R}(\\hat{\\mathbf{x}}) &
            \\text{A measure of the rate of } \\hat{\\mathbf{x}} \\\\
                \\mathrm{R}_{NN} & \\text{The rate of the neural networks}\\\\
            \\mathrm{D}(\hat{\\mathbf{x}}, \\mathbf{x})  & \\text{A distortion
            metric specified by \\texttt{--tune}}
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
        tune: Specify the distortion metrics used to compute the loss.
            Available: "mse", "mse_msssim".
            Defaults to "mse".

    Returns:
        Object gathering the different quantities computed by this loss
        function. Chief among them: the loss itself.
    """

    match tune:
        case "mse":
            mse = _compute_mse(decoded_image, target_image)
            # Compute MS-SSIM only when we want to get the full results.
            # Otherwise, it is not required.
            ms_ssim = (
                _compute_ms_ssim(decoded_image, target_image) if compute_logs else None
            )
            dist = mse
        case "mse_msssim":
            mse = _compute_mse(decoded_image, target_image)
            ms_ssim = _compute_ms_ssim(decoded_image, target_image)
            dist = 0.5 * mse + 0.01 * (1 - ms_ssim)

    if isinstance(decoded_image, Tensor):
        n_pixels = decoded_image.size()[-2] * decoded_image.size()[-1]
    else:
        n_pixels = decoded_image.get("y").size()[-2] * decoded_image.get("y").size()[-1]

    rate_bpp = (rate_latent_bit.sum() + rate_mlp_bit) / n_pixels

    loss = dist + lmbda * rate_bpp

    # Construct the output module, only the loss is always returned
    output = LossFunctionOutput(
        loss=loss,
        mse=mse.detach().item() if (compute_logs and mse is not None) else None,
        ms_ssim=ms_ssim.detach().item() if (compute_logs and ms_ssim is not None) else None,
        rate_nn_bpp=rate_mlp_bit / n_pixels if compute_logs else None,
        rate_latent_bpp=rate_latent_bit.detach().sum().item() / n_pixels
        if compute_logs
        else None,
    )

    return output
