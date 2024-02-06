# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import torch

from typing import TypedDict, Union
from torch import Tensor
from utils.yuv import DictTensorYUV
from encoding_management.loss.ms_ssim import msssim_fn


class DistortionWeighting(TypedDict):
    """Weighting for the different distortions metrics during training."""
    mse: float
    msssim: float
    lpips: float


def compute_mse(x: Union[Tensor, DictTensorYUV], y: Union[Tensor, DictTensorYUV]) -> Tensor:
    """Compute the Mean Squared Error between two images. Both images can either be
    a single tensor, or a dictionary of tensors with one for each color-channel.

    Args:
        x and y (Union[Tensor, DictTensorYUV]): Compute the MSE of x and y.

    Returns:
        Tensor: One element tensor containing the MSE of x and y.
    """
    flag_yuv = not(isinstance(x, Tensor))

    if not flag_yuv:
        return torch.pow((x - y), 2.0).mean()
    else:
        # Total number of pixels for all channels
        total_pixels_yuv = 0.

        # MSE weighted by the number of pixels in each channels
        mse = torch.zeros((1), device=x.get('y').device)
        for (_, x_channel), (_, y_channel) in zip(x.items(), y.items()):
            n_pixels_channel = x_channel.numel()
            mse = mse + torch.pow((x_channel - y_channel), 2.0).mean() * n_pixels_channel
            total_pixels_yuv += n_pixels_channel
        mse = mse / total_pixels_yuv
        return mse

def compute_msssim(x: Union[Tensor, DictTensorYUV], y: Union[Tensor, DictTensorYUV]) -> Tensor:
    """Compute the MS-SSIM between two images. Both images can either be
    a single tensor, or a dictionary of tensors with one for each color-channel.

    Args:
        x and y (Union[Tensor, DictTensorYUV]): Compute the MS-SSIM of x and y.

    Returns:
        Tensor: One element tensor containing the MS-SSIM of x and y.
    """
    flag_yuv = not(isinstance(x, Tensor))

    if not flag_yuv:
        return msssim_fn(
            x,
            y,
            window_size=11,
            size_average=True,
            val_range=None,
            full=True
        )
    else:
        # MS-SSIM weighted by the number of pixels in each channels
        ms_ssim = torch.zeros((1), device=x.get('y').device)
        total_pixels_yuv = 0

        for (_, x_channel), (_, y_channel) in zip(x.items(), y.items()):
            n_pixels_channel = x_channel.numel()
            ms_ssim = ms_ssim + msssim_fn(x_channel, y_channel) * n_pixels_channel
            total_pixels_yuv += n_pixels_channel

        ms_ssim = ms_ssim / total_pixels_yuv
        # Max MS-SSIM dB is 100 i.e. - 10 * log10(1e-10
        ms_ssim = torch.clamp(ms_ssim, 0, 1 - 1e-6)
        return ms_ssim
