#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


"""Compute the bd-rate between two sets of points.

Inspired from:
https://github.com/shengbinmeng/Bjontegaard_metric/blob/master/bjontegaard_metric.py
"""

import argparse
from typing import Optional

import numpy as np
from scipy import interpolate


def check_bdrate_input(rate: np.ndarray, quality: np.ndarray):
    assert len(rate) == len(quality), (
        f"Rate and quality should have identical length. Found {len(rate)} "
        f"elements in rate and {len(quality)} elements in quality."
    )

    assert rate.ndim == 1, (
        f"Rate should be 1-dimensional. Found {rate.ndim} dimensions."
    )
    assert quality.ndim == 1, (
        f"Quality should be 1-dimensional. Found {quality.ndim} dimensions."
    )
    assert rate.min() >= 0, "Rate should have only positive values."
    assert quality.min() >= 0, "Quality should have only positive values."


def compute_bd_rate(
    ref_rate: np.ndarray,
    ref_quality: np.ndarray,
    candidate_rate: np.ndarray,
    candidate_quality: np.ndarray,
    piecewise: int = 1,
) -> Optional[float]:
    """Print the BD-rate of candidate vs ref. If candidate is better than ref,
    the score will be negative e.g.

        ref_rate = np.array([1., 2., 4., 8., 16.])
        candidate_rate = 0.75 * ref_rate

        ref_quality = np.array([25., 26., 27., 28., 29.])
        candidate_quality = ref_quality

        => -25 %

    Args:
        ref_rate (np.array): Rate of the first (reference system)
        ref_quality (np.array): PSNR of the first (reference system)
        candidate_rate (np.array): Rate of the second system
        candidate_quality (np.array): PSNR of the second system
        piecewise (int, optional): Different mode of interpolation. Newer excel
            macro uses piecewise interpolation, while the bdrateOld macro does
            not use piecewise interpolation. Defaults to 1.

    Returns:
        Optional[float]: BD-rate in percent, None if rank warning error
    """
    ref_log_rate = np.log(ref_rate)
    candidate_log_rate = np.log(candidate_rate)

    # rate method
    polynom_ref = np.polyfit(ref_quality, ref_log_rate, 3)
    polynom_candidate = np.polyfit(candidate_quality, candidate_log_rate, 3)

    # integration interval
    min_int = max(min(ref_quality), min(candidate_quality))
    max_int = min(max(ref_quality), max(candidate_quality))

    # find integral
    if piecewise == 0:
        int_polynom_ref = np.polyint(polynom_ref)
        int_polynom_candidate = np.polyint(polynom_candidate)

        int_ref = np.polyval(int_polynom_ref, max_int) - np.polyval(
            int_polynom_ref, min_int
        )
        int_candidate = np.polyval(int_polynom_candidate, max_int) - np.polyval(
            int_polynom_candidate, min_int
        )

    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v_ref = interpolate.pchip_interpolate(
            np.sort(ref_quality), np.sort(ref_log_rate), samples
        )
        v_candidate = interpolate.pchip_interpolate(
            np.sort(candidate_quality), np.sort(candidate_log_rate), samples
        )
        # Calculate the integral using the trapezoid method on the samples.
        int_ref = np.trapz(v_ref, dx=interval)
        int_candidate = np.trapz(v_candidate, dx=interval)

    # find avg diff
    avg_exp_diff = (int_candidate - int_ref) / (max_int - min_int)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100
    return avg_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_quality",
        help="Quality metric of the ref. Coma separated e.g. --ref_quality=20.1,22,24,28.2",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--candidate_quality",
        help="Quality metric of the candidate. "
        "Coma separated e.g. --candidate_quality=20.1,22,24,28.2",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ref_rate",
        help="Rate of the ref. Coma separated e.g. --ref_rate=0.1,0.3,1.2,2.4",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--candidate_rate",
        help="Rate of the candidate. Coma separated e.g. --candidate_rate=0.1,0.3,1.2,2.4",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    def parse_array(arg_array: str) -> np.ndarray:
        return np.array(arg_array.split(",")).astype(np.float64)

    bd_rate = compute_bd_rate(
        parse_array(args.ref_rate),
        parse_array(args.ref_quality),
        parse_array(args.candidate_rate),
        parse_array(args.candidate_quality),
    )
    print(f"{bd_rate:.5f} %")
