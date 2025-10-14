# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import os
from typing import TypedDict, Union

import numpy as np
import torch
import torch.nn.functional as F
from enc.io.types import FRAME_DATA_TYPE, POSSIBLE_BITDEPTH
from enc.utils.device import POSSIBLE_DEVICE
from torch import Tensor


class DictTensorYUV(TypedDict):
    """``TypedDict`` representing a YUV420 frame..

    .. hint::

        ``torch.jit`` requires I/O of modules to be either ``Tensor``, ``List``
        or ``Dict``. So we don't use a python dataclass here and rely on
        ``TypedDict`` instead.

    Args:
        y (Tensor): Tensor with shape :math:`([B, 1, H, W])`.
        u (Tensor): Tensor with shape :math:`([B, 1, \\frac{H}{2}, \\frac{W}{2}])`.
        v (Tensor): Tensor with shape :math:`([B, 1, \\frac{H}{2}, \\frac{W}{2}])`.
    """

    y: Tensor
    u: Tensor
    v: Tensor


def read_yuv(
    file_path: str,
    frame_idx: int,
    frame_data_type: FRAME_DATA_TYPE,
    bit_depth: POSSIBLE_BITDEPTH,
) -> Union[DictTensorYUV, Tensor]:
    """From a file_path /a/b/c.yuv, read the desired frame_index
    and return a dictionary of tensor containing the YUV values:

    .. code:: none

        {
            'Y': [1, 1, H, W],
            'U': [1, 1, H / S, W / S],
            'V': [1, 1, H / S, W / S],
        }

    ``S`` is either 1 (444 sampling) or 2 (420). The YUV values are in [0., 1.]

    Args:
        file_path: Absolute path of the video to load
        frame_idx: Index of the frame to load, starting at 0.
        frame_data_type: chroma sampling (420,444)
        bit depth: Number of bits per component (8 or 10 bits).

    Returns:
        For 420, return a dict of tensors with YUV values of shape [1, 1, H, W].
        For 444 return a [1, 3, H, W] tensor.
    """

    # Parse height and width from the file_path
    w, h = [
        int(tmp_str)
        for tmp_str in os.path.basename(file_path).split(".")[0].split("_")[1].split("x")
    ]

    if frame_data_type == "yuv420":
        w_uv, h_uv = [int(x / 2) for x in [w, h]]
    else:
        w_uv, h_uv = w, h

    # Switch between 8 bit file and 10 bit
    byte_per_value = 1 if bit_depth == 8 else 2

    n_val_y = h * w
    n_val_uv = h_uv * w_uv
    n_val_per_frame = n_val_y + 2 * n_val_uv

    n_bytes_y = n_val_y * byte_per_value
    n_bytes_uv = n_val_uv * byte_per_value
    n_bytes_per_frame = n_bytes_y + 2 * n_bytes_uv

    # Read the required frame and put it in a 1d tensor
    raw_video = torch.tensor(
        np.memmap(
            file_path,
            mode="r",
            shape=n_val_per_frame,
            offset=n_bytes_per_frame * frame_idx,
            dtype=np.uint8 if bit_depth <=8 else np.uint16,
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


@torch.no_grad()
def write_yuv(
    data: Union[Tensor, DictTensorYUV],
    bitdepth: POSSIBLE_BITDEPTH,
    frame_data_type: FRAME_DATA_TYPE,
    file_path: str,
    norm: bool = True,
) -> None:
    """Store a YUV frame as a YUV file named file_path. They are appended to the
    end of the file_path If norm is True: the video data is expected to be in
    [0., 1.] so we multiply it by 255. Otherwise we let it as is.

    Args:
        data: Data to save
        bitdepth: Bitdepth, should be in``[8, 9, 10, 11, 12, 13, 14, 15, 16]``.
        frame_data_type: Data type, either ``"yuv420"`` or ``"yuv444"``.
        file_path: Absolute path of the file where the YUV is saved.
        norm: True to multiply the data by 2 ** bitdepth - 1.
            Defaults to True.
    """
    assert frame_data_type in ["yuv420", "yuv444"], (
        f"Found incorrect datatype in write_yuv() function: {frame_data_type}. "
        'Data type should be "yuv420" or "yuv444".'
    )

    if frame_data_type == "yuv420":
        raw_data = torch.cat([channels.flatten() for _, channels in data.items()])
    else:
        raw_data = data.flatten()

    if norm:
        raw_data = raw_data * (2**bitdepth - 1)

    dtype = np.uint16 if bitdepth == 10 else np.uint8

    # Round the values and cast them to uint8 or uint16 tensor
    raw_data = torch.round(raw_data).cpu().numpy().astype(dtype)

    # Write this to the desired file_path
    np.memmap.tofile(raw_data, file_path)

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def rgb2yuv(rgb: Tensor) -> Tensor:
    """Convert a 4D RGB tensor [1, 3, H, W] into a 4D YUV444 tensor [1, 3, H, W]
    using ITU-R BT.709 coefficients.
    The RGB and YUV values are in the range [0., 1.]

    Args:
        rgb: 4D RGB tensor to convert in [0., 1.]

    Returns:
        The resulting YUV444 tensor in [0. 1.]
    """
    assert (
        len(rgb.size()) == 4
    ), f"rgb2yuv input must be a 4D tensor [B, 3, H, W]. Data size: {rgb.size()}"
    assert (
        rgb.size()[1] == 3
    ), f"rgb2yuv input must have 3 channels. Data size: {rgb.size()}"

    # assert rgb.min() >= 0, "Trying to convert rgb value smaller than 0."

    # assert rgb.max() <= 1, "Trying to convert YUV with data bigger than 1."

    # Split the [1, 3, H, W] into 3 [1, 1, H, W] tensors
    r, g, b = rgb.split(1, dim=1)

    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]

    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    # Concatenate them into the resulting yuv 4D tensor.
    yuv = torch.cat((y, cb, cr), dim=1)
    return yuv


def yuv2rgb(yuv: Tensor):
    """Convert a 4D YUV tensor [1, 3, H, W] into a 4D RGB tensor [1, 3, H, W]
    using ITU-R BT.709 coefficients.
    The RGB and YUV values are in the range [0., 1.]

    Args:
        rgb: 4D YUV444 tensor to convert in [0., 1.]

    Returns:
        The resulting RGB tensor in [0., 1.]
    """
    assert (
        len(yuv.size()) == 4
    ), f"yuv2rgb input must be a 4D tensor [B, 3, H, W]. Data size: {yuv.size()}"
    assert (
        yuv.size()[1] == 3
    ), f"yuv2rgb input must have 3 channels. Data size: {yuv.size()}"

    # assert yuv.min() >= 0, "Trying to convert YCrCb value smaller than 0."

    # assert yuv.max() <= 1, "Trying to convert YCrCb with data bigger than 1."

    y, cb, cr = yuv.split(1, dim=1)

    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]

    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = torch.cat((r, g, b), dim=1)
    return rgb


def yuv_dict_clamp(yuv: DictTensorYUV, min_val: float, max_val: float) -> DictTensorYUV:
    """Clamp the y, u & v tensor.

    Args:
        yuv: The data to clamp
        min_val: Minimum value for the clamp
        max_val: Maximum value for the clamp

    Returns:
        The clamped data

    """
    clamped_yuv = DictTensorYUV(
        y=yuv.get("y").clamp(min_val, max_val),
        u=yuv.get("u").clamp(min_val, max_val),
        v=yuv.get("v").clamp(min_val, max_val),
    )
    return clamped_yuv


def yuv_dict_to_device(yuv: DictTensorYUV, device: POSSIBLE_DEVICE) -> DictTensorYUV:
    """Send a ``DictTensor`` to a device.

    Args:
        yuv: Data to be sent to a device.
        device: The requested device

    Returns:
        Data on the appropriate device.
    """
    return DictTensorYUV(
        y=yuv.get("y").to(device), u=yuv.get("u").to(device), v=yuv.get("v").to(device)
    )


def convert_444_to_420(yuv444: Tensor) -> DictTensorYUV:
    """From a 4D YUV 444 tensor :math:`(B, 3, H, W)`, return a
    ``DictTensorYUV``. The U and V tensors are down sampled using a nearest
    neighbor downsampling.

    Args:
        yuv444: YUV444 data :math:`(B, 3, H, W)`

    Returns:
        YUV420 dictionary of 4D tensors
    """
    assert yuv444.dim() == 4, f"Number of dimension should be 5, found {yuv444.dim()}"

    b, c, h, w = yuv444.size()
    assert c == 3, f"Number of channel should be 3, found {c}"

    # No need to downsample y channel but it should remain a 5D tensor
    y = yuv444[:, 0, :, :].view(b, 1, h, w)

    # Downsample U and V channels together
    uv = F.interpolate(yuv444[:, 1:3, :, :], scale_factor=(0.5, 0.5), mode="nearest")
    u, v = uv.split(1, dim=1)

    yuv420 = DictTensorYUV(y=y, u=u, v=v)
    return yuv420


def convert_420_to_444(yuv420: DictTensorYUV) -> Tensor:
    """Convert a DictTensorYUV to a 4D tensor:math:`(B, 3, H, W)`.
    The U and V tensors are up sampled using a nearest neighbor upsampling

    Args:
        yuv420: YUV420 dictionary of 4D tensor

    Returns:
        YUV444 Tensor :math:`(B, 3, H, W)`
    """
    u = F.interpolate(yuv420.get("u"), scale_factor=(2, 2))
    v = F.interpolate(yuv420.get("v"), scale_factor=(2, 2))
    yuv444 = torch.cat((yuv420.get("y"), u, v), dim=1)
    return yuv444
