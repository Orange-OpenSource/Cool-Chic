# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import math
from torch import Tensor


def mse_fn(x: Tensor, y: Tensor) -> Tensor:
    return (x - y).square().mean()

def dist_to_db(dist: float, max_db: float = 100) -> float:
    min_dist = 10 ** (-max_db / 10)
    dist = max(dist, min_dist)
    return -10 * math.log10(dist)
