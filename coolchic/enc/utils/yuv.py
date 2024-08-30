# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import os

import numpy as np
import torch
from einops import rearrange
from enc.utils.codingstructure import (
    FRAME_DATA_TYPE,
    POSSIBLE_BITDEPTH,
    DictTensorYUV,
    FrameData,
)
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_tensor


def yuv_dict_clamp(yuv: DictTensorYUV, min_val: float, max_val: float) -> DictTensorYUV:
    """Clamp the y, u & v tensor.

    Args:
        yuv (DictTensorYUV): The data to clamp
        min_val (float): Minimum value for the clamp
        max_val (float): Maximum value for the clamp

    Returns:
        DictTensorYUV: The clamped data

    """
    clamped_yuv = DictTensorYUV(
        y=yuv.get("y").clamp(min_val, max_val),
        u=yuv.get("u").clamp(min_val, max_val),
        v=yuv.get("v").clamp(min_val, max_val),
    )
    return clamped_yuv


def load_frame_data_from_file(filename: str, idx_display_order: int) -> FrameData:
    """Load the idx_display_order-th frame from a .yuv file or .png file. For the latter,
    idx_display_order must be equal to 0 as there is only one frame in a png.

    Args:
        filename (str): Absolute path of the file from which the frame is loaded.
        idx_display_order (int): Index of the frame in display order

    Returns:
        FrameData: The loaded frame, wrapped as a FrameData object.
    """

    if filename.endswith(".yuv"):
        # ! We only consider yuv420 and 444 planar
        bitdepth: POSSIBLE_BITDEPTH = 8 if "_8b" in filename else 10
        frame_data_type: FRAME_DATA_TYPE = "yuv420" if "420" in filename else "yuv444"
        data = read_yuv(filename, idx_display_order, frame_data_type, bitdepth)

    elif filename.endswith(".png"):
        bitdepth: POSSIBLE_BITDEPTH = 8
        frame_data_type: FRAME_DATA_TYPE = "rgb"
        data = to_tensor(Image.open(filename))
        data = rearrange(data, "c h w -> 1 c h w")

    return FrameData(bitdepth, frame_data_type, data)


def read_yuv(filename: str, frame_idx: int, frame_data_type: FRAME_DATA_TYPE, bit_depth: POSSIBLE_BITDEPTH) -> DictTensorYUV:
    """From a filename /a/b/c.yuv, read the desired frame_index
    and return a dictionary of tensor containing the YUV values:
        {
            'Y': [1, 1, H, W],
            'U': [1, 1, H / S, W / S],
            'V': [1, 1, H / S, W / S],
        }
    S is either 1 (444 sampling) or 2 (420)
    The YUV values are in [0., 1.]

    Args:
        filename (str): Absolute path of the video to load
        frame_idx (int): Index of the frame to load, starting at 0.
        bit depth (int):number of bits per component (8 or 10 bits).
        frame_data_type chroma sampling (420,444):

    Returns:
        DictTensorYUV: The YUV values (see format above) for 420.
        pytorch tensor for 444 sampling format (consistent with rgb representation)
    """

    # Parse height and width from the filename
    w, h = [
        int(tmp_str)
        for tmp_str in os.path.basename(filename).split(".")[0].split("_")[1].split("x")
    ]

    if frame_data_type == "yuv420":
        w_uv, h_uv = [int(x / 2) for x in [w, h]]
    else:
        w_uv, h_uv = w, h

    # Switch between 8 bit file and 10 bit
    byte_per_value = 1 if bit_depth == 8 else 2

    # We only handle YUV420 for now
    n_val_y = h * w
    n_val_uv = h_uv * w_uv
    n_val_per_frame = n_val_y + 2 * n_val_uv

    n_bytes_y = n_val_y * byte_per_value
    n_bytes_uv = n_val_uv * byte_per_value
    n_bytes_per_frame = n_bytes_y + 2 * n_bytes_uv

    # Read the required frame and put it in a 1d tensor
    raw_video = torch.tensor(
        np.memmap(
            filename,
            mode="r",
            shape=n_val_per_frame,
            offset=n_bytes_per_frame * frame_idx,
            dtype=np.uint16 if bit_depth == 10 else np.uint8,
        ).astype(np.float32)
    )

    # Read the different values from raw video and store them inside y, u and v
    ptr = 0
    y = raw_video[ptr : ptr + n_val_y].view(1, 1, h, w)
    ptr += n_val_y
    u = raw_video[ptr : ptr + n_val_uv].view(1, 1, h_uv, w_uv)
    ptr += n_val_uv
    v = raw_video[ptr : ptr + n_val_uv].view(1, 1, h_uv, w_uv)

    # PyTorch expect data in [0., 1.]; normalize by either 255 or 1023
    norm_factor = 2**bit_depth - 1

    if frame_data_type == "yuv420":
        video = DictTensorYUV(y=y / norm_factor, u=u / norm_factor, v=v / norm_factor)
    else:
        video = torch.cat([y, u, v], dim=1) / norm_factor

    return video


def write_yuv(data: FrameData, filename: str, norm: bool = True) -> None:
    """Store a YUV frame as a YUV file named filename. All parameters of the YUV
    file (resolution, chroma subsampling, bitdepth) are contained in the FrameData
    object alongside the actual data. They are appended to the end of the filename
    If norm is True: the video data is expected to be in [0., 1.] so we
    multiply it by 255. Otherwise we let it as is.

    Args:
        data (FrameData): Data to save
        filename (str): Absolute path of the file where the YUV is saved.
        norm (bool): True to multiply the data by 2 ** bitdepth - 1.
    """
    assert data.frame_data_type in ["yuv420", "yuv444"], (
        "Found incorrect datatype in "
        f'write_yuv() function: {data.frame_data_type}. Data type should be "yuv420" or "yuv444".'
    )

    # Append .yuv at the end of the file to make sure it is present
    if not (filename[-4:] == ".yuv"):
        filename += ".yuv"
    # From here, there is no .yuv at the end of filename
    filename = filename[:-4]

    # Append spatial dimension to the filename, dummy framerate
    # and bit depth
    DUMMY_FRAMERATE = 1
    h, w = data.img_size
    # We need to add a p avec yuv444 otherwise YUView thinks its "YUV444 8-bit packed"
    filename = f"{filename}_{w}x{h}_{DUMMY_FRAMERATE}fps_{data.frame_data_type}p_{data.bitdepth}b.yuv"

    # Concatenate **all** channels into a 2D tensor [1.5 * H * W]
    if data.frame_data_type == "yuv420":
        raw_data = torch.cat([channels.flatten() for _, channels in data.data.items()])
    elif data.frame_data_type == "yuv444":
        raw_data = data.data.flatten()

    if norm:
        raw_data = raw_data * (2**data.bitdepth - 1)

    dtype = np.uint16 if data.bitdepth == 10 else np.uint8

    # Round the values and cast them to uint8 or uint16 tensor
    raw_data = torch.round(raw_data).cpu().numpy().astype(dtype)

    # Write this to the desired filename
    np.memmap.tofile(raw_data, filename)


def rgb2yuv(rgb: Tensor) -> Tensor:
    """Convert a 4D RGB tensor [1, 3, H, W] into a 4D YUV444 tensor [1, 3, H, W].
    The RGB and YUV values are in the range [0, 255]

    Args:
        rgb (Tensor): 4D RGB tensor to convert in [0. 255.]

    Returns:
        Tensor: the resulting YUV444 tensor in [0. 255.]
    """
    assert (
        len(rgb.size()) == 4
    ), f"rgb2yuv input must be a 4D tensor [B, 3, H, W]. Data size: {rgb.size()}"
    assert (
        rgb.size()[1] == 3
    ), f"rgb2yuv input must have 3 channels. Data size: {rgb.size()}"

    # Split the [1, 3, H, W] into 3 [1, 1, H, W] tensors
    r, g, b = rgb.split(1, dim=1)

    # Compute the different channels
    y = torch.round(0.299 * r + 0.587 * g + 0.114 * b)
    u = torch.round(-0.1687 * r - 0.3313 * g + 0.5 * b + +128)
    v = torch.round(0.5 * r - 0.4187 * g - 0.0813 * b + 128)

    # Concatenate them into the resulting yuv 4D tensor.
    yuv = torch.cat((y, u, v), dim=1)
    return yuv


def yuv2rgb(yuv: Tensor):
    """Convert a 4D YUV tensor [1, 3, H, W] into a 4D RGB tensor [1, 3, H, W].
    The RGB and YUV values are in the range [0, 255]

    Args:
        rgb (Tensor): 4D YUV444 tensor to convert in [0. 255.]

    Returns:
        Tensor: the resulting RGB tensor in [0. 255.]
    """
    assert (
        len(yuv.size()) == 4
    ), f"yuv2rgb input must be a 4D tensor [B, 3, H, W]. Data size: {yuv.size()}"
    assert (
        yuv.size()[1] == 3
    ), f"yuv2rgb input must have 3 channels. Data size: {yuv.size()}"

    y, u, v = yuv.split(1, dim=1)
    r = (
        1.0 * y
        + -0.000007154783816076815 * u
        + 1.4019975662231445 * v
        - 179.45477266423404
    )
    g = 1.0 * y + -0.3441331386566162 * u + -0.7141380310058594 * v + 135.45870971679688
    b = (
        1.0 * y
        + 1.7720025777816772 * u
        + 0.00001542569043522235 * v
        - 226.8183044444304
    )
    rgb = torch.cat((r, g, b), dim=1)
    return rgb
