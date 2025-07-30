# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import argparse
import sys

from CCLIB.ccdecapi_cpu import cc_decode_cpu

"""
C++ decode interface
"""

if __name__ == "__main__":
    # =========================== Parse arguments =========================== #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, default="./bitstream.cool", help="Bitstream path."
    )
    parser.add_argument("--output", "-o", default="", help="output ppm (rgb) or yuv")
    parser.add_argument("--no_avx2", action="store_true", help="Disable AVX2 usage")
    parser.add_argument(
        "--verbosity", type=int, default=0,
        help=""
        "0 does not output anything ; "
        "1 prints the runtime of each step ;"
        "2 is for debug."
    )
    parser.add_argument(
        "--output_chroma_format",
        type=int,
        default=0,
        help=
        "Use 0 to infer this from the bitstream header. "
        " "
        "Otherwise, specify '420' or '444' to change the chroma sampling for the "
        "YUV output. "
        " "
        " Useless for RGB."
    )
    parser.add_argument(
        "--output_bitdepth",
        type=int,
        default=0,
        help=
        "Use 0 to infer this from the bitstream header. "
        " "
        "Otherwise, specify an integer in [8, 16] to set the output bitdepth."
    )
    parser.add_argument(
        "--motion_accuracy",
        type=int,
        default=64,
        help="motion accuracy 1/n; n defaults to 64."
    )
    args = parser.parse_args()
    # =========================== Parse arguments =========================== #

    if not args.no_avx2 and sys.platform == "darwin":
        print(
            ""
            "[WARNING]  AVX2 instructions are not available/supported on MAC. "
            "Fallback to normal CPU decoding.\n"
            "You can avoid this Warning by using --no_avx2"
        )

        use_avx2 = False
    else:
        use_avx2 = not args.no_avx2

    if use_avx2:
        from CCLIB.ccdecapi_avx2 import cc_decode_avx2

        if args.verbosity >= 2:
            print("Using AVX2 instructions for faster decoding")
        cc_decode_avx2(
            args.input,
            args.output,
            args.output_bitdepth,
            args.output_chroma_format,
            args.motion_accuracy,
            args.verbosity,
        )
    else:
        cc_decode_cpu(
            args.input,
            args.output,
            args.output_bitdepth,
            args.output_chroma_format,
            args.motion_accuracy,
            args.verbosity,
        )
