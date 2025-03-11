# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import numpy as np
from common.io.png import is_png, read_png, write_png
from common.io.ppm import PPM_POSSIBLE_BITDEPTH, _parse_header_ppm, is_ppm, read_ppm, write_ppm


def is_image(file_path: str) -> bool:
    """Return True if the file is a PNG file or a PPM file

    Args:
        file_path (str): File to be checked

    Returns:
        bool: True if file is a png or ppm
    """

    return is_png(file_path) or is_ppm(file_path)


def read_image(file_path: str) -> np.ndarray:
    """Read an image (either PNG or PPM). Return a numpy array [H, W, 3]
    containing the image data rescaled into [0., 1.].

    Args:
        file_path: Path to the image to be read.

    Returns:
        np.ndarray: Image data.
    """

    assert is_image(file_path), (
        f"read_image is only possible on image. Input file is {file_path}."
    )

    if is_png(file_path):
        return read_png(file_path)

    if is_ppm(file_path):
        return read_ppm(file_path)


def write_image(
    data: np.ndarray, file_path: str, bitdepth: PPM_POSSIBLE_BITDEPTH = 8
) -> None:
    """Write a numpy array as an image file. The numpy array must be
    in [0., 1.] with a shape of [H, W, 3]. The extension of the desired
    file_path (.png or .ppm) determines the output format.

    In the case of .ppm files, the bitdepth can be specified.

    Args:
        data: Numpy array to store in a png file. Values are in
            [0., 1.] and shape is [H, W, 3]
        file_path: Path where the data are stored
        bitdepth: Only for PPM files, desired output bitdepth.
    """

    assert is_image(file_path), (
        f"read_image is only possible on image. Input file is {file_path}."
    )

    if is_png(file_path):
        if bitdepth != 8:
            print(
                f"A bitdepth != 8 is specified (bitdepth={bitdepth}). This is"
                "ignored for PNG files which always have 8-bit data."
            )

        return write_png(data, file_path)

    if is_ppm(file_path):
        return write_ppm(data, file_path, bitdepth=bitdepth)


def get_bitdepth_image(file_path: str) -> PPM_POSSIBLE_BITDEPTH:
    """Return the bitdepth of a given image. In case of PNG, this is always
    8. But it can varies for PPM.

    Args:
        file_path (str): Path of the file whose bitdepth is returned.

    Returns:
        PPM_POSSIBLE_BITDEPTH: List of possible bitdepth
    """
    assert is_image(file_path), (
        f"get_bitdepth_image is only possible on image. Input file is {file_path}."
    )

    # ! Pillow only reads 8-bit png
    if is_png(file_path):
        return 8

    if is_ppm(file_path):
        _, info = _parse_header_ppm(file_path)
        return info["bitdepth"]
