# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

def get_sub_bitstream_path(
    root_bitstream_path: str,
    counter_2d_latent: int,
) -> str:
    """Return the complete path of the sub-bistream corresponding to the
    <counter_2d_latent>-th 2D feature maps. This is use due to the fact that
    even 3D features are coded as independent 2D features.

    Args:
        root_bitstream_path (str): Root name of the bitstream
        counter_2d_latent (int): Index of 2D features. Let us suppose that we have
            two features maps, one is [1, 1, H, W] and the other is [1, 2, H/2, W/2].
            The counter_2d_latent will be used to iterate on the **3** 2d features
            (one for the highest resolution, two for the smallest resolution).

    Returns:
        str: Complete bitstream path
    """
    s = f'{root_bitstream_path}_{counter_2d_latent}'
    return s
