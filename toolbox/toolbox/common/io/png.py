# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import numpy as np
from PIL import Image


def read_png(file_path: str) -> np.ndarray:
    """Read a PNG file and return a numpy array describing it.
    The output shape is [H, W, 3] with values in [0., 1.].

    Args:
        file_path (str): Path of the file to load.

    Returns:
        np.ndarray: Loaded data as a numpy array, shape is [H, W, 3] and
            values are in [0., 1.].
    """
    im = np.array(Image.open(file_path))

    # Remove alpha channel if any, cast to float 32 and divide by 255
    # so that the value are in [0., 1]
    im = im[:, :, :3].astype(np.float32) / 255.0

    return im


def write_png(data: np.ndarray, file_path: str) -> None:
    """Write a numpy array as a PNG file. The numpy array must be
    in [0., 1.] with a shape of [H, W, 3].

    Args:
        data (np.ndarray): Numpy array to store in a png file. Values are in
            [0., 1.] and shape is [H, W, 3]
        file_path (str): Path where the data are stored
    """
    assert len(data.shape) == 3 and data.shape[-1] == 3, (
        f"Data shape must be [H, W, 3], found {data.shape}"
    )

    # Recast the PNG to integer with 256 levels
    data = np.clip(data, 0.0, 1.0)
    data = np.round(data * 255).astype(np.uint8)

    assert data.min() >= 0 and data.max() <= 255, (
        "Data should be in [0., 255.] prior to be saved into PNG file. \n"
        f"Found data.min() = {data.min()} and data.max() = {data.max()}."
    )

    im = Image.fromarray(data)
    im.save(file_path)


def is_png(file_path: str) -> bool:
    """Return True if the file is a PNG file, ending with
    ".png" or ".PNG".

    Args:
        file_path (str): File to be checked

    Returns:
        bool: True if file is a png file
    """

    return file_path.endswith(".png") or file_path.endswith(".PNG")
