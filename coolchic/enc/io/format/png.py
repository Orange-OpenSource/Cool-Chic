# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import os
from typing import Tuple

import torch
from einops import rearrange
from enc.io.types import POSSIBLE_BITDEPTH
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_pil_image, to_tensor


def read_png(file_path: str) -> Tuple[Tensor, POSSIBLE_BITDEPTH]:
    """Read a PNG file

    Args:
        file_path: Path of the png file to read.

    Returns:
        Image data [1, 3, H, W] in [0., 1.] and its bitdepth.
    """

    assert os.path.isfile(file_path), f"No file found at {file_path}"

    data = to_tensor(Image.open(file_path))
    data = rearrange(data, "c h w -> 1 c h w")

    # Bitdepth is always 8 when we read PNG through PIL?
    bitdepth = 8

    return data, bitdepth


@torch.no_grad()
def write_png(data: Tensor, file_path: str) -> None:
    """Save an image x into a PNG file.

    Args:
        x: Image to be saved
        file_path: Where to save the PNG files
    """
    data = rearrange(data, "1 c h w -> c h w", c=3)
    to_pil_image(data).save(file_path)
