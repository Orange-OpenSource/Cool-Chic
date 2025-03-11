#!/usr/bin/env python3


# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


"""Extract the i-th frame of a YUV file and save it somewhere"""

import argparse
import os
import subprocess

from common.io.yuv import get_yuv_info


def extract_frame(
    input_path: str, output_path: str, frame_idx: int, verbosity: int
) -> None:
    """Wrapper around the dd command to extract the i-th frame of a YUV file
    and save it somewhere.

    Args:
        input_path (str): Path of the input video
        output_path (str): Path of the output extracted frame
        frame_idx (int): Index of the frame to be extracted. 0 is the first
        verbosity (int): Set to > 0 to print more stuff
    """
    assert os.path.isfile(input_path), f"File {input_path} not found!"
    assert input_path != output_path, (
        f"Input and output files must be different. Found {input_path} for both"
    )

    yuv_info = get_yuv_info(input_path)

    n_pixel_per_frame = yuv_info.width * yuv_info.height + 2 * (
        yuv_info.width_uv * yuv_info.height_uv
    )
    n_bytes_per_frame = (
        n_pixel_per_frame if yuv_info.bitdepth <= 8 else n_pixel_per_frame * 2
    )

    block_size = n_bytes_per_frame
    blocks_to_skip = frame_idx
    cmd = (
        "dd"
        f" if={input_path}"
        f" of={output_path}"
        f" bs={block_size}"
        f" skip={blocks_to_skip}"
        f" count=1"
    )
    if verbosity == 0:
        cmd += " status=none "

    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Path of the input video", type=str, required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path of the output extracted frame",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--idx", help="Index of the frame to be extracted", type=int, required=True
    )
    parser.add_argument(
        "-v", "--verbosity", help="verbosity level", type=int, default=0
    )
    args = parser.parse_args()

    extract_frame(args.input, args.output, args.idx, args.verbosity)
