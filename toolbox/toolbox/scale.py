#!/usr/bin/env python3


# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import argparse
import math
from typing import Literal, Tuple

import numpy as np
from common.io.image import get_bitdepth_image, is_image, read_image, write_image
from common.io.yuv import YUVData, get_yuv_info, read_one_yuv_frame, write_yuv
from PIL import Image

# https://stackoverflow.com/questions/35381551/fast-interpolation-resample-of-numpy-array-python
FILTER_TRANSLATION = {
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
    "lanczos": Image.Resampling.LANCZOS,
}
FILTER_TYPE = Literal["bicubic", "bilinear", "nearest", "lanczos"]


def resize_yuv_data(
    data: YUVData,
    target_size_y: Tuple[int, int],
    target_size_uv: Tuple[int, int],
    filter_type: FILTER_TYPE = "lanczos",
) -> YUVData:
    res = YUVData(
        y=resize_tensor(data.y, target_size_y, filter_type=filter_type),
        u=resize_tensor(data.u, target_size_uv, filter_type=filter_type),
        v=resize_tensor(data.v, target_size_uv, filter_type=filter_type),
    )

    return res


def resize_tensor(
    data: np.ndarray, target_size: Tuple[int, int], filter_type: FILTER_TYPE = "lanczos"
) -> np.ndarray:
    """Resize a numpy array using Pillow.

    Args:
        data (np.ndarray): Array to be resized. Either [H, W] or [H, W, C].
        target_size (Tuple[int, int]): Target size after resizing. Tuple format
            is (W', H') i.e. desired width and desired height.
        filter_type (FILTER_TYPE, optional): Filter used to resize the tensor.
            Defaults to "lanczos".

    Returns:
        np.ndarray: Resized tensor. Either [H', W'] or [H', W', C].
    """
    assert data.ndim in [2, 3], (
        "Shape of the tensor to resize should be 2-dimensional [Height, Width] "
        f"or 3-dimensional [Height, Width, Channel]. Current tensor is {data.ndim}-"
        f"dimensional, with shape {data.shape}."
    )

    assert len(target_size) == 2, (
        "Target size should be a 2-element Tuple (target width, target height)."
        f" Found: {target_size}."
    )

    # We have a 2D tensor, that is temporarily promoted to a 3D one.
    # This additional dimension is removed at the very end, just before
    # returning the results.
    two_dim_out = data.ndim == 2
    if two_dim_out:
        data = np.expand_dims(data, axis=-1)

    n_channels = data.shape[-1]
    # Reverse target_size because it follows Pillow convention (width, height)
    # not numpy convention (height, width).
    res = np.empty((target_size[1], target_size[0], n_channels), dtype=data.dtype)

    # Do the upsampling channel by channel as Image.fromarray does not seem to
    # work with [H, W, 3], np.float32 tensors with values in [0., 1.]
    for c in range(n_channels):
        im = Image.fromarray(data[:, :, c])
        im_resize = im.resize(target_size, resample=FILTER_TRANSLATION[filter_type])
        res[:, :, c] = np.array(im_resize, dtype=np.float32)

    # Some filters (e.g. lanczos or bicubic) can overshoot.
    res = np.clip(res, 0.0, 1.0)

    if two_dim_out:
        res = res[:, :, 0]

    return res


def compute_target_size(
    initial_height: int, initial_width: int, scale: float
) -> Tuple[int, int]:
    """Compute the target size [Width, Height] (in pixels) given the current
    tensor size and the desired scale. In the case the required scaling does not
    gives an entire number of pixels, round to the closest number e.g.
    a [67, 64] tensor with scale=0.25 gives a [17, 16] target size.

    Args:
        initial_height int: Height of the array to be resized.
        initial_width int: width of the array to be resized.
        scale (float): Relative scaling to be applied. 0.5 means downsampling
            by a factor of two.

    Returns:
        Tuple[int, int]: Target size after resizing. Tuple format is (W', H')
        i.e. desired width. and desired height.
    """

    assert scale > 0, f"--scale must be > 0. Found --scale={scale}"

    target_size = tuple(
        [int(math.floor(tmp * scale)) for tmp in [initial_width, initial_height]]
    )
    return target_size


def parse_target_size_args(size: str) -> Tuple[int, int]:
    """From a <width>x<height> string, return a tuple of int (width, height).

    Args:
        size (str): Argument --size=<width>x<height>

    Returns:
        Tuple[int, int]: The same width and height, put into a tuple of int.
    """
    return tuple([int(tmp) for tmp in size.split("x")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file", type=str)
    parser.add_argument("-o", "--output", help="Output file", type=str)
    parser.add_argument(
        "--filter",
        help="Filter used for chroma up/down sampling. "
        f"Available: {list(FILTER_TRANSLATION.keys())}",
        type=str,
        default="lanczos",
    )

    # Two exclusive arguments: specify either the scaling required or the
    # target output size
    group = parser.add_argument_group(
        "Target resolution", "Specify the required output size"
    )
    exclusive_group = group.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument(
        "--scale",
        help="Scaling factor: 2 means output height and width is two times bigger "
        "than the input. 0.5 means output is smaller. "
        "-1 means that we don't use this argument but the --size one.",
        type=float,
        default=-1,
    )

    exclusive_group.add_argument(
        "--size",
        help="Target output size. Format: <width>x<height>."
        "For YUV 420 data, this size refers to the Y channels"
        "An empty string means that we don't use this argument but the --scale one.",
        type=str,
        default="",
    )
    args = parser.parse_args()

    assert args.scale != -1 or args.size != "", (
        "At least one of --scale or --size must be provided!"
    )

    if is_image(args.input):
        im = read_image(args.input)

        if args.scale != -1:
            height, width, _ = im.shape
            target_size = compute_target_size(
                initial_height=height, initial_width=width, scale=args.scale
            )
        else:
            target_size = parse_target_size_args(args.size)

        scaled_im = resize_tensor(im, target_size, filter_type=args.filter)

        # Same output and input bitdepth
        write_image(
            scaled_im,
            args.output,
            bitdepth=get_bitdepth_image(args.input)

        )

    else:
        yuv_info = get_yuv_info(args.input)

        if args.scale != -1:
            target_size_y = compute_target_size(
                initial_height=yuv_info.height,
                initial_width=yuv_info.width,
                scale=args.scale,
            )

            if yuv_info.chroma_sampling == "420":
                assert (target_size_y[0] % 2 == 0) and (target_size_y[1] % 2 == 0), (
                    "For YUV 420 data, --target_size must be a multiple of 2, "
                    "so that the U and V channels have an entire number of pixels. "
                    f"Found target_size={target_size_y} due to scale = {args.scale}"
                )

            target_size_uv = compute_target_size(
                initial_height=yuv_info.height_uv,
                initial_width=yuv_info.width_uv,
                scale=args.scale,
            )
        else:
            # Look at the desired target size for Y, deduce the scale
            # recompute the UV target size.
            target_size_y = parse_target_size_args(args.size)

            if yuv_info.chroma_sampling == "420":
                assert (target_size_y[0] % 2 == 0) and (target_size_y[1] % 2 == 0), (
                    "For YUV 420 data, --target_size must be a multiple of 2, "
                    "so that the U and V channels have an entire number of pixels. "
                    f"Found --target_size={target_size_y}"
                )

            # There can be a different scale for height and width as we don't
            # have to respect aspect ratio.
            scale_y_width = target_size_y[0] / yuv_info.width
            scale_y_height = target_size_y[1] / yuv_info.height

            # Compute target_size twice, once for the width (using scale_width)
            # and once for the height using scale_height.
            target_size_uv_w, _ = compute_target_size(
                initial_height=yuv_info.height_uv,
                initial_width=yuv_info.width_uv,
                scale=scale_y_width,
            )
            _, target_size_uv_h = compute_target_size(
                initial_height=yuv_info.height_uv,
                initial_width=yuv_info.width_uv,
                scale=scale_y_height,
            )
            target_size_uv = (target_size_uv_w, target_size_uv_h)

        # Rescale the successive frame
        for frame_idx in range(yuv_info.n_frames):
            original_frame = read_one_yuv_frame(args.input, yuv_info, frame_idx)
            rescaled_frame = resize_yuv_data(
                original_frame, target_size_y, target_size_uv, filter_type=args.filter
            )
            write_yuv(
                rescaled_frame,
                args.output,
                yuv_info.bitdepth,
                # Write for the first frame, append for the others
                mode="w" if frame_idx == 0 else "a",
            )
