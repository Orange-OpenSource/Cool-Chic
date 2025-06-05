#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import argparse
import typing
from typing import Union

import numpy as np
from common.io.image import get_bitdepth_image, is_image, read_image, write_image
from common.io.yuv import (
    COLORSPACE,
    YUVData,
    get_dense_array,
    get_yuv_info,
    read_one_yuv_frame,
    write_yuv,
)

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def _bt709_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """Convert a YUV 444 image into a RGB image.
    Using ITU-R BT.709 coefficients.

    Args:
        ycbcr (np.ndarray): YUV (actually YCbCr?) data to be converted
            to RGB. Shape is [H, W, 3]

    Returns:
        np.ndarray: Dense array representing the RGB values. Output shape
            is [H, W, 3].
    """
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]

    y, cb, cr = [
        # All channels have a shape of [H, W, 1]
        np.expand_dims(ycbcr[:, :, i], axis=-1)
        for i in range(3)
    ]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = np.concatenate([r, g, b], axis=-1)

    return rgb


def _rgb_to_bt709(rgb: np.ndarray) -> np.ndarray:
    """Convert a RGB image into a YUV (YCbCr ?) image.
    Using ITU-R BT.709 coefficients.

    Args:
        ycbcr (np.ndarray): RGB data to be converted to YUV. Shape is [H, W, 3]

    Returns:
        np.ndarray: Dense array representing the YUV values. Output shape
            is [H, W, 3].
    """
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]

    r, g, b = [
        # All channels have a shape of [H, W, 1]
        np.expand_dims(rgb[:, :, i], axis=-1)
        for i in range(3)
    ]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    ycbcr = np.concatenate([y, cb, cr], axis=-1)

    return ycbcr


def _YCoCg2rgb(y, Co, Cg):
    """YCoCg to RGB conversion f

    Args:
        y,cb,cr :
    Returns:
        rgb : converted
    """
    g = y + Cg
    tmp = y - Cg
    r = tmp + Co
    b = tmp - Co

    return [r, g, b]


def _rgb2YCoCg(r, g, b):
    """RGB to YCoCg conversion for torch Tensor.

    Args:r,g,b

    Returns:
        YCoCg : converted
    """

    y = 0.25 * r + 0.50 * g + 0.25 * b
    Co = 0.50 * r - 0.50 * b
    Cg = -0.25 * r + 0.50 * g - 0.25 * b

    return [y, Co, Cg]


def color_transform(
    x: Union[np.ndarray, YUVData], from_color: COLORSPACE, to_color: COLORSPACE
) -> np.ndarray:
    """Transform data from one color space to an other.
    If the original data format is YUV, its chroma sampling must be 444.
    Use the color_sampling.py script to upsample the chroma channels if needed.

    Args:
        x (Union[np.ndarray, YUVData]): Input data to transform.
        from_color (COLORSPACE): Original color space.
        to_color (COLORSPACE): Destination color space.

    Returns:
        np.ndarray: Dense array [H, W, 3] representing the data in a new
            color space.
    """
    if isinstance(x, YUVData):
        x = get_dense_array(x)

    if from_color == to_color:
        return x

    # All the available color transforms are summed up in this dictionary
    COLOR_TRANSFORM = {
        "bt709_to_rgb": _bt709_to_rgb,
        "rgb_to_bt709": _rgb_to_bt709,
    }

    key_transform = f"{from_color}_to_{to_color}"
    assert key_transform in COLOR_TRANSFORM, (
        f"Unknown transform from {from_color} to {to_color}. "
        f"Available: {COLOR_TRANSFORM.keys()}"
    )
    transform = COLOR_TRANSFORM.get(key_transform)

    return transform(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file", type=str)
    parser.add_argument("-o", "--output", help="Output file", type=str)
    parser.add_argument(
        "--from_color",
        help=(f"Initial color domain. Available: {typing.get_args(COLORSPACE)}"),
        type=str,
    )
    parser.add_argument(
        "--to_color",
        help=(f"Target color domain. Available: {typing.get_args(COLORSPACE)}"),
        type=str,
    )
    args = parser.parse_args()

    assert args.from_color in typing.get_args(
        COLORSPACE
    ) and args.to_color in typing.get_args(COLORSPACE), (
        f"--from_color and --to_color must be in {typing.get_args(COLORSPACE)}.\n"
        f"Found --from_color={args.from_color} and --to_color={args.to_color}"
    )

    if is_image(args.input):
        n_frames = 1

    # Input is not png --> it is YUV
    else:
        input_yuv_info = get_yuv_info(args.input)
        n_frames = input_yuv_info.n_frames

        if is_image(args.output):
            assert n_frames == 1, (
                "It is not possible to store the results of a color "
                f"transform from a multi-frame YUV file to an image file {args.output}. "
                "The input YUV file must have a single frame."
            )

    # Apply the color transform on the successive frames
    for frame_idx in range(n_frames):
        if is_image(args.input):
            original_frame = read_image(args.input)
        else:
            original_frame = read_one_yuv_frame(args.input, input_yuv_info, frame_idx)

        transformed_frame = color_transform(
            original_frame, from_color=args.from_color, to_color=args.to_color
        )

        if is_image(args.output):
            # Same output and input bitdepth
            write_image(
                transformed_frame,
                args.output,
                bitdepth=get_bitdepth_image(args.input)
            )

        # Output is YUV
        else:
            output_yuv_info = get_yuv_info(args.output)
            # Cosmetic for now, might change later.
            output_yuv_info.colorspace = args.to_color
            output_yuv_frame = YUVData(
                y=transformed_frame[:, :, 0],
                u=transformed_frame[:, :, 1],
                v=transformed_frame[:, :, 2],
            )

            write_yuv(
                output_yuv_frame,
                args.output,
                output_yuv_info.bitdepth,
                # Write for the first frame, append for the others
                mode="w" if frame_idx == 0 else "a",
            )
