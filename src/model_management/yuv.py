# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import os
import subprocess
import torch
import numpy as np
import torch.nn.functional as F

from typing import Optional, Tuple, TypedDict
from torch import Tensor


class DictTensorYUV(TypedDict):
    """How we represent a YUV420 frame."""
    y: Tensor       # [1, 1, H, W]
    u: Tensor       # [1, 1, H / 2, W / 2]
    v: Tensor       # [1, 1, H / 2, W / 2]


def yuv_dict_to_device(yuv: DictTensorYUV, device: str) -> DictTensorYUV:
    """Send the y, u & v channel of a DictTensorYUV to a device.

    Args:
        yuv (DictTensorYUV): The data to send to a device
        device (str): Either "cpu", "cuda:0"

    Returns:
        DictTensorYUV: The data sent to a device
    """

    return DictTensorYUV(
        y=yuv.get('y').to(device), u=yuv.get('u').to(device), v=yuv.get('v').to(device)
    )


def yuv_dict_clamp(yuv: DictTensorYUV, min_val: float, max_val: float) -> DictTensorYUV:
    """Clamp the y, u & v tensor.

    Args:
        yuv (DictTensorYUV): The data to clamps
        min_val (float): Minimum value for the clamp
        max_val (float): Maximum value for the clamp

    Returns:
        DictTensorYUV: The clamped data

    """
    clamped_yuv = DictTensorYUV(
        y=yuv.get('y').clamp(min_val, max_val),
        u=yuv.get('u').clamp(min_val, max_val),
        v=yuv.get('v').clamp(min_val, max_val)
    )
    return clamped_yuv


def read_video(filename: str, frame_idx: int) -> DictTensorYUV:
    """From a filename /a/b/c.yuv, read the desired frame_index
    and return a dictionary of tensor containing the YUV values:
        {
            'Y': [1, 1, H, W],
            'U': [1, 1, H / 2, W / 2],
            'V': [1, 1, H / 2, W / 2],
        }
    The YUV values are in [0., 1.]

    /!\ bit depth and resolution are inferred from the filename which should
        be something like:
            B-MarketPlace_1920x1080_60p_yuv420_10b.yuv


    Args:
        filename (str): Absolute path of the video to load
        frame_idx (int): Index of the frame to load, starting at 0.

    Returns:
        DictTensorYUV: The YUV values (see format above).
    """

    # Parse height and width from the filename
    w, h = [int(tmp_str) for tmp_str in os.path.basename(filename).split(".")[0].split("_")[1].split("x")]
    w_uv, h_uv = [int(x / 2) for x in [w, h]]

    # Switch between 8 bit file and 10 bit
    bit_depth = 8 if 'yuv420_8b' in filename else 10
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
            mode='r',
            shape=(n_val_per_frame),
            offset=n_bytes_per_frame * frame_idx,
            dtype=np.uint16 if bit_depth == 10 else np.uint8
        ).astype(np.float32)
    )

    # Read the different values from raw video and store them inside y, u and v
    ptr = 0
    y = raw_video[ptr: ptr + n_val_y].view(1, 1, h, w)
    ptr += n_val_y
    u = raw_video[ptr: ptr + n_val_uv].view(1, 1, h_uv, w_uv)
    ptr += n_val_uv
    v = raw_video[ptr: ptr + n_val_uv].view(1, 1, h_uv, w_uv)

    # PyTorch expect data in [0., 1.]; normalize by either 255 or 1023
    norm_factor = 2 ** bit_depth - 1
    video = DictTensorYUV(y=y / norm_factor, u=u / norm_factor, v=v / norm_factor)
    return video


def convert_444_to_420(yuv444: Tensor) -> DictTensorYUV:
    """From a 4D tensor [B, 3, H, W], return a DictTensorYUV (see format above)
    The U and V tensors are down sampled using a linear down sampling

    Args:
        frame (Tensor): YUV444 4D tensor

    Returns:
        DictTensorYUV: YUV420 dictionary of 4D tensor
    """
    assert yuv444.dim() == 4, f'Number of dimension should be 5, found {yuv444.dim()}'

    b, c, h, w = yuv444.size()
    assert c == 3, f'Number of channel should be 3, found {c}'

    # No need to downsample y channel but it should remain a 5D tensor
    y = yuv444[:, 0, :, :].view(b, 1, h, w)

    # Downsample U and V channels together
    uv = F.interpolate(
        yuv444[:, 1:3, :, :], scale_factor=(0.5, 0.5), mode='nearest'
    )
    u, v = uv.split(1, dim=1)

    yuv420 = DictTensorYUV(y=y, u=u, v=v)
    return yuv420


def write_yuv(yuv420: DictTensorYUV, filename: str, norm: bool = True, bitdepth: int = 8):
    """Store a YUV dictionary of 4D tensor as a YUV file named filename.
    The resolution of the YUV file is inferred from data and appended to the
    end of the filename.
    If norm is True: the video data is expected to be in [0., 1.] so we
    multiply it by 255. Otherwise we let it as is.

    Args:
        yuv420 (DictTensorYUV): Data to save (see format above)
        filename (str): Absolute path of the file where the YUV is saved.
        norm (bool): True to multiply the data by 2 ** bitdepth - 1.
        bitdepth (int): Either 8 or 10
    """

    # Append .yuv at the end of the file to make sure it is present
    if not(filename[-4:] == '.yuv'):
        filename += '.yuv'
    # From here, there is no .yuv at the end of filename
    filename = filename[:-4]

    # Append spatial dimension to the filename, dummy framerate
    # and bit depth
    _, _, h, w = yuv420.get('y').size()
    DUMMY_FRAMERATE = 1
    filename = f'{filename}_{w}x{h}_{DUMMY_FRAMERATE}p_yuv420_{bitdepth}b.yuv'

    # Concatenate **all** channels into a 2D tensor [1.5 * H * W]
    raw_data = torch.cat(
        [yuv420[k].reshape(-1) for k in ['y', 'u', 'v']], dim=0
    )

    if norm:
        raw_data = raw_data * (2 ** bitdepth - 1)

    dtype = np.uint16 if bitdepth == 10 else np.uint8

    # Round the values and cast them to uint8 or uint16 tensor
    raw_data = torch.round(raw_data.flatten()).cpu().numpy().astype(dtype)

    # Write this to the desired filename
    np.memmap.tofile(raw_data, filename)


def write_y4m(data: Tensor, path_save: str, mode: str = 'yuv444_8b', h_w: Optional[Tuple[int, int]]=None):
    """Save a tensor as a .yuv file. It will be flatten with data.flatten()

    Args:
        data (Tensor): image to be saved
        path_save (str): absolute path where the file is saved
        mode (str): 'yuv420_8b', 'yuv420_10b', 'yuv444_8b', 'yuv444_10b'
        h_w (tuple): height and width of the Y channel. If omitted, it is deduced from data.size()
    """
    # Write the .y4m header as a additional file
    if h_w is None:
        h, w = [str(int(x)) for x in data.size()[-2:]]
    else:
        h, w = h_w

    with open(f'{path_save}.header', 'wb') as f_out:
        header = b'YUV4MPEG2 '
        header += f'W{w} '.encode('ascii')
        header += f'H{h} '.encode('ascii')
        header += 'F30:1 '.encode('ascii')
        header += b'A1:1 '
        header += b'Ip '
        if '444' in mode:
            header += b'C444'
        elif '420' in mode:
            header += b'C420'
        header += b'\x0A'
        header += 'FRAME'.encode('ascii')
        header += b'\x0A'
        f_out.write(header)

    # Flatten all the channels of the tensor and save the 8-bit values
    # into an additional .data file
    dtype = np.uint8 if '8b' in mode else np.uint16
    data.flatten().numpy().astype(dtype).tofile(f'{path_save}.data')

    # Append both header file and data file to the final file
    subprocess.call(f'cat {path_save}.header > {path_save}', shell=True)
    subprocess.call(f'cat {path_save}.data >> {path_save}', shell=True)

    # Delete the intermediate files
    subprocess.call(f'rm {path_save}.header {path_save}.data', shell=True)

def rgb2yuv(rgb: Tensor) -> Tensor:
    """Convert a 4D RGB tensor [1, 3, H, W] into a 4D YUV444 tensor [1, 3, H, W].
    The RGB and YUV values are in the range [0, 255]

    Args:
        rgb (Tensor): 4D RGB tensor to convert.

    Returns:
        Tensor: the resulting YUV444 tensor
    """
    # Split the [1, 3, H, W] into 3 [1, 1, H, W] tensors
    r, g, b = rgb.split(1, dim=1)

    # Compute the different channels
    y = torch.round(0.299 * r + 0.587 * g + 0.114 * b)
    u = torch.round(-0.1687 * r - 0.3313 * g + 0.5 * b +  + 128)
    v = torch.round(0.5 * r - 0.4187 * g - 0.0813 * b + 128)

    # Concatenate them into the resulting yuv 4D tensor.
    yuv = torch.cat((y, u, v), dim=1)
    return yuv
