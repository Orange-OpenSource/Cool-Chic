#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import argparse
import typing
from typing import Literal

import numpy as np
from common.io.image import is_image
from common.io.yuv import (
    CHROMA_SAMPLING,
    YUVData,
    check_420,
    check_444,
    get_yuv_info,
    read_one_yuv_frame,
    write_yuv,
)
from PIL import Image

# https://stackoverflow.com/questions/35381551/fast-interpolation-resample-of-numpy-array-python
FILTER_TRANSLATION = {
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
    "lanczos": Image.Resampling.LANCZOS,
}

FILTER_TYPE = Literal["bicubic", "bilinear", "nearest", "lanczos"]


def convert_444_to_420(
    yuv_data: YUVData, filter_type: FILTER_TYPE = "bicubic"
) -> YUVData:
    """Convert 444 YUV to 420 YUV.

    Args:
        yuv_data (YUVData): 444 Data
        filter_type (FILTER_TYPE, optional): Filter used to downsampled the
            chroma channels. Defaults to "bicubic".

    Returns:
        YUVData: 420 Data
    """
    assert check_444(yuv_data), (
        "Trying to convert from 444 to 420 while input format is not 444."
        f"Resolutions: Y = {yuv_data.y.shape} ; "
        f"U = {yuv_data.u.shape} ; V = {yuv_data.v.shape}."
    )

    # PIL Image are indexed with [width, height], unlike numpy tensors which
    # are [height, width].
    y_height, y_width = yuv_data.y.shape
    chroma_new_res = [y_width // 2, y_height // 2]
    im = Image.fromarray(yuv_data.u)
    ud = im.resize(chroma_new_res, resample=FILTER_TRANSLATION[filter_type])
    ud = np.array(ud, dtype=np.float32)

    im = Image.fromarray(yuv_data.v)
    vd = im.resize(chroma_new_res, resample=FILTER_TRANSLATION[filter_type])
    vd = np.array(vd, dtype=np.float32)

    # Do we need np.array() to create a copy and avoid changing yuv_data
    # in place?
    yuv_data_420 = YUVData(y=np.array(yuv_data.y), u=ud, v=vd)

    return yuv_data_420


def convert_420_to_444(yuv_data: YUVData, filter_type: FILTER_TYPE = "bicubic"):
    """Convert 420 YUV to 444 YUV.

    Args:
        yuv_data (YUVData): 420 Data
        filter_type (FILTER_TYPE, optional): Filter used to upsampled the
            chroma channels. Defaults to "bicubic".

    Returns:
        YUVData: 420 Data
    """
    assert check_420(yuv_data), (
        "Trying to convert from 420 to 444 while input format is not 420."
        f"Resolutions: Y = {yuv_data.y.shape} ; "
        f"U = {yuv_data.u.shape} ; V = {yuv_data.v.shape}."
    )

    # PIL Image are indexed with [width, height], unlike numpy tensors which
    # are [height, width].
    y_height, y_width = yuv_data.y.shape
    chroma_new_res = [y_width, y_height]
    im = Image.fromarray(yuv_data.u)
    ud = im.resize(chroma_new_res, resample=FILTER_TRANSLATION[filter_type])
    ud = np.array(ud, dtype=np.float32)

    im = Image.fromarray(yuv_data.v)
    vd = im.resize(chroma_new_res, resample=FILTER_TRANSLATION[filter_type])
    vd = np.array(vd, dtype=np.float32)

    # Do we need np.array() to create a copy and avoid changing yuv_data
    # in place?
    yuv_data_444 = YUVData(y=np.array(yuv_data.y), u=ud, v=vd)

    return yuv_data_444


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input yuv file", type=str)
    parser.add_argument("-o", "--output", help="Output yuv file", type=str)
    parser.add_argument(
        "--out_format",
        help=f"Output chroma format. Available: {typing.get_args(CHROMA_SAMPLING)}",
        type=str,
    )
    parser.add_argument(
        "--filter",
        help="Filter used for chroma up/down sampling. "
        f"Available: {list(FILTER_TRANSLATION.keys())}",
        type=str,
        default="bicubic",
    )

    args = parser.parse_args()

    assert args.out_format in typing.get_args(CHROMA_SAMPLING), (
        f"--out_format must be in {CHROMA_SAMPLING}. Found {args.out_format}"
    )

    assert args.filter in FILTER_TRANSLATION.keys(), (
        f"--out_format must be in {FILTER_TRANSLATION.keys()}. Found {args.out_format}."
    )

    assert not is_image(args.input), (
        f"Color sampling is only available for yuv file. Found --input={args.input}"
    )

    yuv_info = get_yuv_info(args.input)

    # Resample the chroma channels of the successive frames
    for frame_idx in range(yuv_info.n_frames):
        original_frame = read_one_yuv_frame(args.input, yuv_info, frame_idx)

        if args.out_format == "420":
            output_frame = convert_444_to_420(original_frame, filter_type=args.filter)
        elif args.out_format == "444":
            output_frame = convert_420_to_444(original_frame, filter_type=args.filter)

        write_yuv(
            output_frame,
            args.output,
            yuv_info.bitdepth,
            # Write for the first frame, append for the others
            mode="w" if frame_idx == 0 else "a",
        )
