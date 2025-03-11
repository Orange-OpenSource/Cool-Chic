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


def mse_fn(x: np.ndarray, y: np.ndarray) -> float:
    """Return the mean squared error (MSE) between two arrays."""
    return np.mean((x - y) ** 2)


def mse_to_psnr(mse: float) -> float:
    """Convert a mean squared error (MSE) to a peak signal to noise ratio
    (PSNR). The MSE is assumed to have been computed on data in the
    [0., 1.] range.

    Note: the maximal PSNR is 100 dB as we add a 1e-10 constant to the MSE.

    Args:
        mse (float): Mean squared Error

    Returns:
        float: PSNR in dB
    """
    return -10 * np.log10(mse + 1e-10)


def psnr_to_mse(psnr: float) -> float:
    """Convert a peak signal to noise ratio (PSNR) into a mean squared error
    (MSE). The MSE is assumed to have been computed on data in the
    [0., 1.] range.

    Args:
        PSNR (float): Peak signal to noise ratio in dB

    Returns:
        float: Mean squared error
    """
    return 10 ** (-psnr / 10)


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

        psnr = mse_to_psnr(mse_fn(candidate, ref))
        results = {
            "candidate": args.candidate,
            "PSNR RGB (dB)": f"{psnr:7.4f}",
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

            mse_y = mse_fn(ref_frame.y, candidate_frame.y)
            mse_u = mse_fn(ref_frame.u, candidate_frame.u)
            mse_v = mse_fn(ref_frame.v, candidate_frame.v)

            PSNR_y = mse_to_psnr(mse_y)
            PSNR_u = mse_to_psnr(mse_u)
            PSNR_v = mse_to_psnr(mse_v)

            metric_list.append(
                {
                    "mse_y": mse_y,
                    "mse_u": mse_u,
                    "mse_v": mse_v,
                    "PSNR_y": PSNR_y,
                    "PSNR_u": PSNR_u,
                    "PSNR_v": PSNR_v,
                }
            )

            if args.verbosity:
                print(f"{PSNR_y:7.4f}\t{PSNR_u:7.4f}\t{PSNR_v:7.4f}")

        JVET_PSNR_y = sum(m["PSNR_y"] for m in metric_list) / len(metric_list)
        JVET_PSNR_u = sum(m["PSNR_u"] for m in metric_list) / len(metric_list)
        JVET_PSNR_v = sum(m["PSNR_v"] for m in metric_list) / len(metric_list)
        JVET_WPSNR = (6.0 * JVET_PSNR_y + JVET_PSNR_u + JVET_PSNR_v) / 8.0

        results = {
            "candidate": args.candidate,
            "JVET_WPSNR  (dB)": f"{JVET_WPSNR:7.4f}",
            "JVET_PSNR_y (dB)": f"{JVET_PSNR_y:7.4f}",
            "JVET_PSNR_u (dB)": f"{JVET_PSNR_u:7.4f}",
            "JVET_PSNR_v (dB)": f"{JVET_PSNR_v:7.4f}",
        }

        print(dict_to_str(results, noheader=args.noheader))
