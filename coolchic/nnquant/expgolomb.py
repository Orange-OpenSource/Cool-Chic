# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import math
from dataclasses import fields
from typing import List, Tuple

import torch
from torch import Tensor, nn

from coolchic.component.core.coolchic import DescriptorCoolChic
from coolchic.component.core.types import DescriptorNN

POSSIBLE_EXP_GOL_COUNT = DescriptorCoolChic(
    arm=DescriptorNN(
        weight=torch.linspace(0, 12, 13, device="cpu", dtype=torch.int32),
        bias=torch.linspace(0, 12, 13, device="cpu", dtype=torch.int32),
    ),
    ifce=DescriptorNN(
        weight=torch.linspace(0, 12, 13, device="cpu", dtype=torch.int32),
        bias=torch.linspace(0, 12, 13, device="cpu", dtype=torch.int32),
    ),
    upsampling=DescriptorNN(
        weight=torch.linspace(0, 12, 13, device="cpu", dtype=torch.int32),
        bias=torch.linspace(0, 12, 13, device="cpu", dtype=torch.int32),
    ),
    synthesis=DescriptorNN(
        weight=torch.linspace(0, 12, 13, device="cpu", dtype=torch.int32),
        bias=torch.linspace(0, 12, 13, device="cpu", dtype=torch.int32),
    ),
)
