# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


"""Gather different custom data type."""

from typing import TypedDict, Union


class DescriptorNN(TypedDict, total=False):
    """Contains information (scale, weight, quantization step, ...) about the
    weights and biases of a neural network."""
    weight: Union[int, float, str]
    bias: Union[int, float, str]


class DescriptorCoolChic(TypedDict, total=False):
    """Contains information about the different sub-networks of Cool-chic."""
    arm: DescriptorNN
    upsampling: DescriptorNN
    synthesis: DescriptorNN
