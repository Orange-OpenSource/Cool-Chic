#!/usr/bin/env python3


# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import argparse
import os

import numpy as np
from common.cliprint import dict_to_str
from common.io.image import is_image, read_image
from common.io.yuv import get_yuv_info, read_one_yuv_frame
from scipy import signal
from scipy.ndimage import convolve

# ------ START: CODE FROM GITHUB
# Code from https://github.com/tallamjr/clic/blob/master/metrics/msssim.py

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start : stop, offset + start : stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def _SSIMForMultiScale(
    img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
):
    """Return the Structural Similarity Map between `img1` and `img2`.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).

    Returns:
      Pair containing the mean SSIM and contrast sensitivity between `img1` and
      `img2`.

    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError(
            "Input images must have the same shape (%s vs. %s).", img1.shape, img2.shape
        )
    if img1.ndim != 4:
        raise RuntimeError("Input images must have four dimensions, not %d", img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode="valid")
        mu2 = signal.fftconvolve(img2, window, mode="valid")
        sigma11 = signal.fftconvolve(img1 * img1, window, mode="valid")
        sigma22 = signal.fftconvolve(img2 * img2, window, mode="valid")
        sigma12 = signal.fftconvolve(img1 * img2, window, mode="valid")
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs


def MultiScaleSSIM(
    img1,
    img2,
    max_val=255,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    weights=None,
):
    """Return the MS-SSIM score between `img1` and `img2`.

    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
      weights: List of weights for each level; if none, use five levels and the
        weights from the original paper.

    Returns:
      MS-SSIM score between `img1` and `img2`.

    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError(
            "Input images must have the same shape (%s vs. %s).", img1.shape, img2.shape
        )
    if img1.ndim != 4:
        raise RuntimeError("Input images must have four dimensions, not %d", img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2,
        )
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [
            convolve(im, downsample_filter, mode="reflect") for im in [im1, im2]
        ]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
    return np.prod(mcs[0 : levels - 1] ** weights[0 : levels - 1]) * (
        mssim[levels - 1] ** weights[levels - 1]
    )


# ------ END: CODE FROM GITHUB


def ms_ssim_to_log(ms_ssim: float) -> float:
    """Convert a MS-SSIM to a log scale using the following computation:

    MS-SSIM dB = -10 * log10(1 - MS-SSIM + eps)

    Note: the maximal log MS-SSIM is 100 dB as we add a eps=1e-10
    constant inside the log10.

    Args:
        ms_ssim (float): MS-SSIM

    Returns:
        float: MS-SSIM in dB
    """
    return -10 * np.log10(1 - ms_ssim + 1e-10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", help="Path of the image to evaluate", type=str, required=True
    )
    parser.add_argument(
        "--ref", help="Path of the reference image", type=str, required=True
    )
    parser.add_argument(
        "--noheader", help="add title field to results", action="store_true"
    )
    parser.add_argument(
        "-v", "--verbosity", help="verbosity level", type=int, default=0
    )
    args = parser.parse_args()

    if not os.path.exists(args.ref):
        print(f"File {args.ref} does not exist")
        exit(1)
    if not os.path.exists(args.candidate):
        print(f"File {args.candidate} does not exist")
        exit(1)

    # Check that both the candidate and the references have the same format
    if is_image(args.candidate):
        assert is_image(args.ref), (
            "Both candidate and reference images must be image files. Found "
            f"{args.candidate} and {args.ref}"
        )
    else:
        assert not is_image(args.ref), (
            "Both candidate and reference images must be YUV files. Found "
            f"{args.candidate} and {args.ref}"
        )

    if is_image(args.candidate):
        candidate = read_image(args.candidate)
        ref = read_image(args.ref)
        h_c, w_c, c_c = candidate.shape
        h_r, w_r, c_r = ref.shape

        assert h_c == h_r and w_c == w_r and c_c == c_r, (
            "Shape must be identical between the candidate and "
            f"the reference. Candidate shape is {candidate.shape} and "
            f"reference shape is {ref.shape}"
        )

        # MS-SSIM requires a 4D tensor [1, H, W, C]
        candidate = np.expand_dims(candidate, axis=0)
        ref = np.expand_dims(ref, axis=0)

        ms_ssim_db = ms_ssim_to_log(MultiScaleSSIM(candidate, ref, max_val=1.0))
        results = {
            "candidate": args.candidate,
            "MS-SSIM RGB (dB)": f"{ms_ssim_db:7.4f}",
        }
        print(dict_to_str(results, noheader=args.noheader))

    else:
        yuv_info_ref = get_yuv_info(args.ref)
        yuv_info_candidate = get_yuv_info(args.candidate)

        identical_size = (
            yuv_info_candidate.width == yuv_info_ref.width
            and yuv_info_candidate.height == yuv_info_ref.height
            and yuv_info_candidate.width_uv == yuv_info_ref.width_uv
            and yuv_info_candidate.height_uv == yuv_info_ref.height_uv
        )

        assert identical_size, (
            "Shape must be identical between the candidate and "
            f"the reference. Candidate shape is {get_yuv_info(args.candidate)} and "
            f"reference shape is {get_yuv_info(args.ref)}"
        )

        metric_list = []
        for frame_idx in range(yuv_info_candidate.n_frames):
            candidate_frame = read_one_yuv_frame(
                args.candidate, yuv_info_ref, frame_idx
            )
            ref_frame = read_one_yuv_frame(args.ref, yuv_info_ref, frame_idx)

            # MS-SSIM requires a 4D tensor [1, H, W, C]
            ms_ssim_y = MultiScaleSSIM(
                img1=np.expand_dims(ref_frame.y, axis=(0, -1)),
                img2=np.expand_dims(candidate_frame.y, axis=(0, -1)),
                max_val=1.0,
            )
            ms_ssim_u = MultiScaleSSIM(
                img1=np.expand_dims(ref_frame.u, axis=(0, -1)),
                img2=np.expand_dims(candidate_frame.u, axis=(0, -1)),
                max_val=1.0,
            )
            ms_ssim_v = MultiScaleSSIM(
                img1=np.expand_dims(ref_frame.v, axis=(0, -1)),
                img2=np.expand_dims(candidate_frame.v, axis=(0, -1)),
                max_val=1.0,
            )

            ms_ssim_db_y = ms_ssim_to_log(ms_ssim_y)
            ms_ssim_db_u = ms_ssim_to_log(ms_ssim_u)
            ms_ssim_db_v = ms_ssim_to_log(ms_ssim_v)

            metric_list.append(
                {
                    "ms_ssim_y": ms_ssim_y,
                    "ms_ssim_u": ms_ssim_u,
                    "ms_ssim_v": ms_ssim_v,
                    "ms_ssim_db_y": ms_ssim_db_y,
                    "ms_ssim_db_u": ms_ssim_db_u,
                    "ms_ssim_db_v": ms_ssim_db_v,
                }
            )

            if args.verbosity:
                print(f"{ms_ssim_db_y:7.4f}\t{ms_ssim_db_u:7.4f}\t{ms_ssim_db_v:7.4f}")

        overall_ms_ssim_y = sum(m["ms_ssim_y"] for m in metric_list) / len(metric_list)
        overall_ms_ssim_u = sum(m["ms_ssim_u"] for m in metric_list) / len(metric_list)
        overall_ms_ssim_v = sum(m["ms_ssim_v"] for m in metric_list) / len(metric_list)

        n_pixel_y = ref_frame.y.size
        n_pixel_u = ref_frame.u.size
        n_pixel_v = ref_frame.v.size
        total_pixel = n_pixel_y + n_pixel_u + n_pixel_v
        overall_ms_ssim = (
            n_pixel_y * overall_ms_ssim_y
            + n_pixel_u * overall_ms_ssim_u
            + n_pixel_v * overall_ms_ssim_v
        ) / total_pixel

        overall_ms_ssim_db_y = ms_ssim_to_log(overall_ms_ssim_y)
        overall_ms_ssim_db_u = ms_ssim_to_log(overall_ms_ssim_u)
        overall_ms_ssim_db_v = ms_ssim_to_log(overall_ms_ssim_v)
        overall_ms_ssim_db = ms_ssim_to_log(overall_ms_ssim)

        results = {
            "candidate": args.candidate,
            "MS-SSIM (dB)": f"{overall_ms_ssim_db:7.4f}",
            "MS-SSIM_y (dB)": f"{overall_ms_ssim_db_y:7.4f}",
            "MS-SSIM_u (dB)": f"{overall_ms_ssim_db_u:7.4f}",
            "MS-SSIM_v (dB)": f"{overall_ms_ssim_db_v:7.4f}",
        }

        print(dict_to_str(results, noheader=args.noheader))
