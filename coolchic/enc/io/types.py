# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from typing import Literal


FRAME_DATA_TYPE = Literal["rgb", "yuv420", "yuv444", "flow"]
POSSIBLE_BITDEPTH = Literal[8, 9, 10, 11, 12, 13, 14, 15, 16]
