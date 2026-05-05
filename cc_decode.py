# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import argparse

from coolchic.bitstream.decode import decode_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, help="Bitstream path.")
    parser.add_argument("--output", "-o", type=str, help="Decoded file path.")
    parser.add_argument("--verbosity", type=int, help="Verbosity level.", default=0)
    args = parser.parse_args()

    decode_video(args.input, decoded_path=args.output, verbosity=args.verbosity)
