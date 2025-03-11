# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import math
import os
from typing import Dict, Literal, Tuple

import numpy as np

PPM_POSSIBLE_BITDEPTH = Literal[8, 9, 10, 11, 12, 13, 14, 15, 16]


def _skip_one_byte(data: bytearray) -> bytearray:
    """Skip one byte in a byte array and return the array
    with its first byte removed.

    Args:
        data: Input byte array. Length is N

    Returns:
        bytearray: Output byte array. Length is N - 1
    """
    return data[1:]


def _read_int_until_blank(data: bytearray) -> Tuple[int, bytearray]:
    """Parse an ASCII int until running into one of the space characters.
    Also return the input byte array where the bytes corresponding to
    both the int and the space character are skipped.

        132\ncds# --> Return the value 123 and a byte array containing cds#


    As defined by the ANSI standard C isspace(s) returns True for
    Horizontal tab (HT), line feed (LF), vertical tabulation (VT),
    form feed (FF), carriage return (CR) and white space.

    Note: this function may fail if the bytes collected up to the space
    character do not represent ascii numbers.

    Args:
        data: Input data.

    Returns:
        Parsed int + input data where we've removed the bytes corresponding to
        the int value and the space character.
    """
    # As defined by the ANSI standard C isspace(s)
    # ASCII code for Horizontal tab (HT), line feed (LF),
    # vertical tabulation (VT), form feed (FF), carriage return (CR)
    # and white space
    _BLANKS_ASCII = [9, 10, 11, 12, 13, 32]

    ptr_end = 0
    while data[ptr_end] not in _BLANKS_ASCII:
        ptr_end += 1

    value = int(data[:ptr_end].decode("utf-8"))
    data = data[ptr_end:]
    return value, data


def _16bits_byte_swap(data: np.ndarray) -> np.ndarray:
    """Invert the bytes composing a 2-byte value. The actual data type
    of the array is not important but it must contains value in
    [0, 2 ** 16 - 1]

    For instance:

            1111 1111 0000 0010 ==> 0000 0010 1111 1111
            \______/  \______/      \______/  \______/
              MSB       LSB            LSB       MSB

    Args:
        data: array to be swapped.

    Returns:
        Swapped array.
    """

    msb = data // 2**8
    lsb = data % 2**8
    swapped_data = lsb * 2**8 + msb
    return swapped_data


def _parse_header_ppm(file_path: str) -> Tuple[np.ndarray, Dict[str, int]]:
    assert os.path.isfile(file_path), f"No file found at {file_path}."

    data = open(file_path, "rb").read()
    magic_number = data[:2].decode("utf-8")
    data = data[2:]
    assert magic_number == "P6", (
        f"Invalid file format. PPM file should start with P6. Found {magic_number}."
    )

    # Parse the header
    width, data = _read_int_until_blank(_skip_one_byte(data))
    height, data = _read_int_until_blank(_skip_one_byte(data))
    max_val, data = _read_int_until_blank(_skip_one_byte(data))
    data = _skip_one_byte(data)

    bitdepth = int(math.log2(max_val + 1))

    info = {
        "width": width,
        "height": height,
        "bitdepth": bitdepth
    }

    return data, info


def read_ppm(file_path: str) -> np.ndarray:
    """Read a `PPM file <https://netpbm.sourceforge.net/doc/ppm.html>`_,
    and return a np array [H, W, 3] containing the data.
    The returned array is in [0., 1.] regardless of the bitdepth.

    .. attention::

        We don't filter out comments inside PPM files...

    Args:
        file_path: Path of the ppm file to read.

    Returns:
        Image data [H, W, 3] in [0., 1.]
    """
    data, info = _parse_header_ppm(file_path)

    width = info.get("width")
    height = info.get("height")
    bitdepth = info.get("bitdepth")

    n_bytes_per_val = 1 if bitdepth <= 8 else 2

    raw_value = np.frombuffer(
        data,
        count=3 * width * height,
        dtype=np.uint8 if n_bytes_per_val == 1 else np.uint16,
    )

    # Re-arrange the value from R1 B1 G1 R2 B2 G2 to an usual [H, W, 3] array
    img = np.empty((height, width, 3), dtype=np.float32)
    for i in range(3):
        img[:, :, i] = raw_value[i::3].reshape(1, height, width)

    # In a PPM file 2-byte value (e.g. 257) is represented as
    # 1111 1111 0000 0010
    # \______/  \______/
    #   MSB       LSB
    # We want to invert these two bytes here to have an usual binary value
    # 0000 0010 1111 1111
    if n_bytes_per_val == 2:
        img = _16bits_byte_swap(img)

    # Normalize in [0. 1.]
    img = img / (2**bitdepth - 1)

    return img


def write_ppm(
    data: np.ndarray, file_path: str, bitdepth: PPM_POSSIBLE_BITDEPTH
) -> None:
    """Save an image x into a PPM file.

    Args:
        data: Image to be saved, must be in [0, 1.]
        bitdepth: Bitdepth, should be in
            ``[8, 9, 10, 11, 12, 13, 14, 15, 16]``.
        file_path: Where to save the PPM files
        bitdepth: Bitdepth of the file. Defaults to 8.
    """
    # Remove all first dimensions of size 1
    assert len(data.shape) == 3, (
        f"write_ppm expects a [H, W, C] data. Found shape {data.shape}"
    )

    assert data.min() >= 0 and data.max() <= 1., (
        f"write_ppm expects data in [0., 1.]. Found data.min() = {data.min()} "
        f"data.max() = {data.max()}."
    )

    h, w, c = data.shape[-3:]

    max_val = 2**bitdepth - 1
    n_bytes_per_val = 1 if max_val <= 255 else 2
    header = f"P6\n{w} {h}\n{max_val}\n"

    data = np.round(data * (2**bitdepth - 1))

    # In a PPM file 2-byte value (e.g. 257) is represented as
    # 1111 1111 0000 0010
    # \______/  \______/
    #   MSB       LSB
    # We want to invert these two bytes here to have an usual binary value
    # 0000 0010 1111 1111
    if n_bytes_per_val == 2:
        data = _16bits_byte_swap(data)

    # Format data as expected by the PPM file.
    flat_data = np.empty((c * h * w), dtype=np.uint8 if max_val <= 255 else np.uint16)
    for i in range(c):
        flat_data[i::3] = data[:, :, i].flatten()

    # Write once the header as a string then the data as binary bytes
    with open(file_path, "w") as f_out:
        f_out.write(header)
    with open(file_path, "ab") as f_out:
        f_out.write(np.memmap.tobytes(flat_data))


def is_ppm(file_path: str) -> bool:
    """Return True if the file is a PPM file, ending with
    ".ppm" or ".PPM".

    Args:
        file_path (str): File to be checked

    Returns:
        bool: True if file is a ppm file
    """

    return file_path.endswith(".ppm") or file_path.endswith(".ppm")
