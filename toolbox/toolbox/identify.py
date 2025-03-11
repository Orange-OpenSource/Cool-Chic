#!/usr/bin/env python3


# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import argparse
import os
from dataclasses import fields
from typing import Any, Dict

from common.cliprint import dict_to_str
from common.io.image import get_bitdepth_image, is_image, read_image
from common.io.yuv import get_yuv_info


def identify_fn(file_path: str) -> Dict[str, Any]:
    info = {}
    if is_image(file_path):
        height, width, _ = read_image(file_path).shape

        # Order here must match the order of the (first) attributes
        # of the YUVDescriptor dataclass so that identify returns
        # identically ordered results for yuv and png files
        info["width"] = width
        info["height"] = height
        info["n_frames"] = 1
        info["bitdepth"] = get_bitdepth_image(file_path)

    # yuv
    else:
        # Put everything from yuv info into info, except color space
        yuv_info = get_yuv_info(file_path)
        for f in fields(yuv_info):
            if f.name == "colorspace":
                pass
            info[f.name] = getattr(yuv_info, f.name)

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Path of the image to identify", type=str, required=True
    )

    parser.add_argument(
        "--noheader", help="add column names to results", action="store_true"
    )
    args = parser.parse_args()

    assert os.path.exists(args.input), f"Can not found a file named {args.input}"

    info = identify_fn(args.input)
    print(dict_to_str(info, noheader=args.noheader))
