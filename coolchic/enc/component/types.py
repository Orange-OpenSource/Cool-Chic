# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from dataclasses import dataclass
from typing import Literal, Optional, Union


@dataclass
class DescriptorNN:
    """Contains information (scale, weight, quantization step, ...) about the
    weights and biases of a neural network."""

    weight: Optional[Union[int, float, str]] = None
    bias: Optional[Union[int, float, str]] = None


@dataclass
class DescriptorCoolChic:
    """Contains information about the different sub-networks of Cool-chic."""

    arm: Optional[DescriptorNN] = None
    upsampling: Optional[DescriptorNN] = None
    synthesis: Optional[DescriptorNN] = None


# For now, it is only possible to have a Cool-chic encoder
# with this name i.e. this key in frame_encoder.coolchic_enc
NAME_COOLCHIC_ENC = Literal["residue", "motion"]

