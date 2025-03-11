# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

CHROMA_SAMPLING = Literal["420", "444"]
BITDEPTH = Literal[8, 10]

# Allow to embed rgb values inside a .yuv file.
# That could be convenient!
COLORSPACE = Literal["bt709", "ycocg", "rgb"]


@dataclass
class YUVDescriptor:
    """Store info about a YUV file."""

    # The order of these fields are not to be changed!
    # The first ones must be the one already describing
    # a non-yuv file (i.e. PNG), so that the identify.py
    # script return info with identical order for both
    # yuv and png.
    width: int
    height: int
    n_frames: int
    bitdepth: BITDEPTH
    width_uv: int
    height_uv: int
    chroma_sampling: CHROMA_SAMPLING
    colorspace: COLORSPACE


@dataclass
class YUVData:
    """Represent the actual data of a YUV file."""

    y: np.ndarray
    u: np.ndarray
    v: np.ndarray


def get_yuv_info(file_path: str) -> Optional[YUVDescriptor]:
    """Parse the name of a .yuv file to extract some information.
    We use the following naming convention:

        /a/b/[name]_[width]x[height]_[fps]p_yuv[chroma_sampling]_[bitdepth]b.yuv

    Args:
        file_path (str): Path of the file, named according to the naming
            convention above.

    Returns:
        Optional[YUVDescriptor]: Info about the file
    """

    # format is Validation03_1920x1080_p25_yuv420_8b.yuv for automatic format detection

    match_format = re.search(r"\/?([a-zA-Z0-9-]+)_(\d+)x(\d+).*.yuv$", file_path)
    if match_format:
        shortname = match_format.groups(1)[0]
        w = int(match_format.groups(1)[1])
        h = int(match_format.groups(1)[2])

    m = re.search(r"_([0-9]+)x([0-9]+)_", file_path)
    if not m:
        print("yuv name mis-formatted: we need the resolution _wxh_ inside the name")
        return 0, 0, 0

    w = int(m.group(1))
    h = int(m.group(2))

    bitdepth = 8
    m = re.search(r"_10b", file_path)
    if m:
        bitdepth = 10

    colorspace = "bt709"
    m = re.search(r"ycocg", file_path)
    if m:
        colorspace = "ycocg"

    chroma_sampling = "444"
    m = re.search(r"420", file_path)
    if m:
        chroma_sampling = "420"

    if chroma_sampling == "420":
        w_uv, h_uv = [x // 2 for x in [w, h]]
    else:
        w_uv, h_uv = [w, h]

    # Switch between 8-bit file and 10-bit file
    byte_per_value = 1 if bitdepth == 8 else 2

    n_val_y = h * w
    n_val_uv = h_uv * w_uv
    n_bytes_y = n_val_y * byte_per_value
    n_bytes_uv = n_val_uv * byte_per_value
    n_bytes_per_frame = n_bytes_y + 2 * n_bytes_uv

    if os.path.isfile(file_path):
        n_frames = os.path.getsize(file_path) / n_bytes_per_frame
        assert float.is_integer(n_frames), (
            f"Found {n_frames} frames for the {file_path}.\nShould be an integer number"
        )
        n_frames = int(n_frames)
    else:
        n_frames = 0

    yuv_descriptor = YUVDescriptor(
        width=w,
        height=h,
        width_uv=w_uv,
        height_uv=h_uv,
        chroma_sampling=chroma_sampling,
        bitdepth=bitdepth,
        colorspace=colorspace,
        n_frames=n_frames,
    )

    return yuv_descriptor


def read_one_yuv_frame(
    file_path: str, yuv_descriptor: YUVDescriptor, frame_idx: int = 0
) -> YUVData:
    """From a file_path /a/b/c.yuv, read the desired frame and return
    its value

    The returned YUV values are always in [0., 1.], regardless of the bitdepth

    Args:
        file_path (str): Absolute path of the video to load
        frame_idx (int, Optional): Index of the frame to load, starting at 0.
            Default to 0.

    Returns:
        YUVData: The YUV values (see format above).
    """

    w, h = yuv_descriptor.width, yuv_descriptor.height
    chroma_sampling = yuv_descriptor.chroma_sampling
    bitdepth = yuv_descriptor.bitdepth

    if chroma_sampling == "420":
        w_uv, h_uv = [x // 2 for x in [w, h]]
    else:
        w_uv, h_uv = [w, h]

    # Switch between 8-bit file and 10-bit file
    byte_per_value = 1 if bitdepth == 8 else 2

    n_val_y = h * w
    n_val_uv = h_uv * w_uv
    n_val_per_frame = n_val_y + 2 * n_val_uv

    n_bytes_y = n_val_y * byte_per_value
    n_bytes_uv = n_val_uv * byte_per_value
    n_bytes_per_frame = n_bytes_y + 2 * n_bytes_uv
    dtype = np.uint16 if bitdepth != 8 else np.uint8
    # Read the required frame and put it in a 1d tensor
    raw_video = np.array(
        np.memmap(
            file_path,
            mode="r",
            shape=n_val_per_frame,
            offset=n_bytes_per_frame * frame_idx,
            dtype=dtype,
        )
    ).astype(np.float32)

    # For 8-bit images & video, the intensity 255 corresponds to 1.
    raw_video = raw_video / (2**bitdepth - 1)

    # Read the different values from raw video and store them inside y, u and v
    ptr = 0
    y = raw_video[ptr : ptr + n_val_y].reshape(h, w)
    ptr += n_val_y
    u = raw_video[ptr : ptr + n_val_uv].reshape(h_uv, w_uv)
    ptr += n_val_uv
    v = raw_video[ptr : ptr + n_val_uv].reshape(h_uv, w_uv)

    # # ! Should I do that?
    # if bitdepth == 10:
    #     norm_factor = 4.0
    #     y /= norm_factor
    #     u /= norm_factor
    #     v /= norm_factor

    # # ! Why? Change to bitdepth?
    # cb = u - 128.0
    # cr = v - 128.0

    return YUVData(y=y, u=u, v=v)


def write_yuv(
    yuv_data: YUVData,
    file_path: str,
    bitdepth: BITDEPTH,
    mode: Literal["w", "a"],
) -> None:
    """Store YUV data into a file.

    Args:
        yuv_data (YUVData): Data to store
        file_path (str): Where to store them
        bitdepth (BITDEPTH): Required to indicate how many bytes are used
            to store each value.
        mode (Literal): "w" to overwrite existing file, "a" to append to the
            end of an existing file.
    """
    raw_data = np.concatenate(
        [
            yuv_data.y.flatten(),
            yuv_data.u.flatten(),
            yuv_data.v.flatten(),
        ]
    )

    # Scale back from [0., 1.] to [0, 2 ** bitdepth - 1]
    raw_data = raw_data * (2**bitdepth - 1)
    raw_data = np.round(raw_data).astype(np.uint8 if bitdepth == 8 else np.uint16)

    max_clip = 2**bitdepth - 1
    assert raw_data.min() >= 0, (
        f"Trying to write YUV file with data smaller than 0. Found {raw_data.min()}"
    )

    assert raw_data.max() < 2**bitdepth, (
        f"Trying to write {bitdepth}-bit YUV file with data bigger than {max_clip}."
        f"Found {raw_data.max()}."
    )

    assert mode in ["w", "a"], f"Unknown mode. Found {mode}, should be in ['w', 'a']"

    # Write the current frame to a temporary file
    tmp_path_cur_frame = f"{file_path}.curframe"
    np.memmap.tofile(raw_data, tmp_path_cur_frame)

    # Cat it (either appending or erasing) to the final file
    append_or_erase = ">" if mode == "w" else ">>"
    cmd = f"cat {tmp_path_cur_frame} {append_or_erase} {file_path}"
    subprocess.call(cmd, shell=True)

    # Remove the temporary file
    os.remove(tmp_path_cur_frame)


def check_444(yuv_data: YUVData) -> bool:
    """Return True if yuv_data is 444 i.e. if Y, U and V have the same
    spatial dimension.

    Args:
        yuv_data (YUVData): Data we want to check

    Returns:
        bool: True if input is 444
    """
    y_res = yuv_data.y.shape
    u_res = yuv_data.u.shape
    v_res = yuv_data.v.shape

    return y_res == u_res and y_res == v_res


def check_420(yuv_data: YUVData) -> bool:
    """Return True if yuv_data is 420 i.e. if U and V spatial dimension
    is half the one the Y channel

    Args:
        yuv_data (YUVData): Data we want to check

    Returns:
        bool: True if input is 420
    """
    y_res = yuv_data.y.shape
    u_res = yuv_data.u.shape
    v_res = yuv_data.v.shape

    normal_uv_shape = tuple([x // 2 for x in y_res])

    return u_res == normal_uv_shape and v_res == normal_uv_shape


def get_dense_array(yuv_data: YUVData) -> np.ndarray:
    """Convert a YUVData object with 3 attributes (y, u and v) to
    a single numpy array. The shape of this numpy array depends on the
    format.

        * YUV444 frame gives a np.array with a shape of [H, W, 3].

        * YUV420 frame gives a np.array with a shape of [1.5 * H * W]


    Args:
        yuv_data (YUVData): YUV data to transform to a dense array.

    Returns:
        np.ndarray: Dense array representing the YUV data.
    """

    if check_444(yuv_data):
        y, u, v = [
            # All channels have a shape of [H, W, 1]
            np.expand_dims(getattr(yuv_data, channel), axis=-1)
            for channel in ["y", "u", "v"]
        ]

        yuv_dense = np.concatenate([y, u, v], axis=-1)

    elif check_420(yuv_data):
        yuv_dense = np.concatenate(
            [yuv_data.y.flatten(), yuv_data.u.flatten(), yuv_data.v.flatten()],
            axis=-1,
        )

    return yuv_dense
