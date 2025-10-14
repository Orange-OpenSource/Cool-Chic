# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import typing
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Union

import torch
from enc.io.format.yuv import DictTensorYUV
from enc.training.metrics.mse import dist_to_db, mse_fn
from enc.training.metrics.wasserstein import wasserstein_fn
from torch import Tensor

DISTORTION_METRIC = Literal["mse", "wasserstein"]

@dataclass(kw_only=True)
class LossFunctionOutput():
    """Output for FrameEncoder.loss_function"""
    # ----- This is the important output
    # Optional to allow easy inheritance by FrameEncoderLogs
    # but will never be None
    loss: Optional[float] = None                                        # The RD cost to optimize
    dist: Optional[float] = None                                        # The distorsion cost to optimize along with the rate
    rate_bpp:Optional[float] = None

    # Any other data required to compute some logs, stored inside a dictionary
    detailed_dist: Optional[Dict[DISTORTION_METRIC, float]] = None      # Each distortion value (mse, wasserstein...)
    rate_latent_bpp: Optional[float] = None                             # Rate associated to the latent          [bpp]
    total_rate_nn_bpp : float = 0.                                      # Total rate associated to the all NNs of all cool-chic [bpp]

    # ==================== Not set by the init function ===================== #
    # Everything here is derived from the above metrics
    total_rate_latent_bpp: Optional[float] = field(init=False, default=None)    # Overall rate of all the latents [bpp]
    dist_db: Optional[float] = None
    detailed_dist_db: Optional[Dict[DISTORTION_METRIC, float]] = field(
        init=False, default_factory=lambda: {}
    )           # Each distortion value (mse, wasserstein...) in dB
    total_rate_bpp: Optional[float] = field(init=False, default=None)           # Overall rate: latent & NNs      [bpp]
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        # Compute some dB values from distortion
        if self.detailed_dist is not None:
            self.detailed_dist_db["psnr_db"] = dist_to_db(self.detailed_dist["mse"])
            if "wasserstein" in self.detailed_dist:
                self.detailed_dist_db["wd_db"] = dist_to_db(self.detailed_dist["wasserstein"])

        self.dist_db = dist_to_db(self.dist)

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
        return mse_fn(x, y)
    else:
        # Total number of pixels for all channels
        total_pixels_yuv = 0.0

        # MSE weighted by the number of pixels in each channels
        mse = torch.zeros((1), device=x.get("y").device)
        for (_, x_channel), (_, y_channel) in zip(x.items(), y.items()):
            n_pixels_channel = x_channel.numel()
            mse = mse + mse_fn(x_channel, y_channel) * n_pixels_channel
            total_pixels_yuv += n_pixels_channel
        mse = mse / total_pixels_yuv
        return mse


def _compute_wasserstein(
    decoded_img: Union[Tensor, DictTensorYUV], target_img: Union[Tensor, DictTensorYUV]
) -> Tensor:
    """Compute the Wasserstein distance between two images. Both images can
    either be a single tensor, or a dictionary of tensors with one for each
    color channel. In case of images with multiple channels, the final Wasserstein
    distance is obtained by averaging the Wasserstein distance for each color channel,
    weighted by the number of pixels. E.g. for YUV 420:
        WD  = (4 * WD_Y + WD_U + WD_V) / 6

    Args:
        x (Union[Tensor, DictTensorYUV]): One of the two inputs
        y (Union[Tensor, DictTensorYUV]): The other input

    Returns:
        Tensor: One element tensor containing the WD of x and y.
    """
    flag_420 = not (isinstance(decoded_img, Tensor))

    if not flag_420:
        wd = wasserstein_fn(decoded_img, target_img)
    else:
        # Total number of pixels for all channels
        total_pixels_yuv = 0.0

        # WD weighted by the number of pixels in each channels
        wd = torch.zeros((1), device=decoded_img.get("y").device)
        for (_, decoded_channel), (_, target_channel) in zip(
            decoded_img.items(), target_img.items()
        ):
            n_pixels_channel = decoded_channel.numel()
            wd = wd + wasserstein_fn(decoded_channel, target_channel) * n_pixels_channel
            total_pixels_yuv += n_pixels_channel
        wd = wd / total_pixels_yuv
    return wd


def loss_function(
    decoded_image: Union[Tensor, DictTensorYUV],
    rate_latent_bit: Dict[str, Tensor],
    target_image: Union[Tensor, DictTensorYUV],
    dist_weight: Dict[DISTORTION_METRIC, float],
    lmbda: float = 1e-3,
    total_rate_nn_bit: float = 0.0,
    compute_logs: bool = False,
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
            metric specified by \\texttt{--tune} and \\texttt{--alpha}}
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

    flag_yuv420 = not isinstance(decoded_image, Tensor)

    device = decoded_image.get("y").device if flag_yuv420 else decoded_image.device

    all_dists = {}
    final_dist = torch.zeros((1), device=device)
    # Iterate on all possible distortion metrics.
    for dist_name, dist_w in dist_weight.items():

        if dist_name == "mse":
            cur_dist = _compute_mse(decoded_image, target_image)
        elif dist_name == "wasserstein":
            cur_dist = _compute_wasserstein(decoded_image, target_image)
        else:
            raise ValueError(
                f"Unsupported distortion metrics. Found {dist_name}, available "
                f"values are {typing.get_args(DISTORTION_METRIC)}. Exiting!"
            )

        all_dists[dist_name] = cur_dist
        # Aggregate weighted dist
        final_dist = final_dist + dist_w * cur_dist

    if flag_yuv420:
        n_pixels = decoded_image.get("y").size()[-2] * decoded_image.get("y").size()[-1]
    else:
        n_pixels = decoded_image.size()[-2] * decoded_image.size()[-1]

    total_rate_latent_bit = torch.cat(
        [v.sum().view(1) for _, v in rate_latent_bit.items()]
    ).sum()
    rate_bpp = total_rate_latent_bit + total_rate_nn_bit
    rate_bpp = rate_bpp / n_pixels

    loss = final_dist + lmbda * rate_bpp

    # Construct the output module, only the loss is always returned
    rate_latent_bpp = None
    total_rate_nn_bpp = 0.0


    if compute_logs:
        rate_latent_bpp = {
            k: v.detach().sum().item() / n_pixels for k, v in rate_latent_bit.items()
        }
        total_rate_nn_bpp = total_rate_nn_bit / n_pixels

        # Detach all distortions only when computing logs
        for k, v in all_dists.items():
            all_dists[k] = v.detach().item()

    output = LossFunctionOutput(
        loss=loss,
        dist=final_dist.detach().item(),
        rate_bpp=rate_bpp.detach().item(),
        detailed_dist=all_dists if compute_logs else None,
        total_rate_nn_bpp=total_rate_nn_bpp,
        rate_latent_bpp=rate_latent_bpp,
    )

    return output
