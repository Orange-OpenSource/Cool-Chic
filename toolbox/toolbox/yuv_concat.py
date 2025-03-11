#!/usr/bin/env python3


# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


"""Extract the i-th frame of a YUV file and save it somewhere"""

import argparse
import subprocess
from typing import List


def concat_frames(input_path: List[str], output_path: str) -> None:
    """Concatenate a list of yuv files to a single one. The frame order
    follows the ordering of the list of paths.

    Args:
        input_path (List[str]): Path of the yuv files to be concatenated
        output_path (str): Path of the resulting yuv file
    """
    assert input_path, "You must provide at least one input_path"

    cmd = "cat "
    for input_path_i in input_path:
        cmd += f"{input_path_i} "
    cmd += f" > {output_path}"

    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path of the input videos to concatenate, separated by comas. E.g"
        " /path/to/videoA.yuv,/path/to/videoB.yuv",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path of the output concatenated video",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    concat_frames(args.input.split(","), args.output)
