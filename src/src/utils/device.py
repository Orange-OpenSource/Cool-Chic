# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

from typing import Literal
import torch

POSSIBLE_DEVICE = Literal['cpu', 'cuda:0', 'mps:0']

def get_best_device() -> POSSIBLE_DEVICE:
    """Return the best available device i.e. best ranked one in the following list:
            1. cuda:0
            2. mps:0
            3. cpu

    Returns:
        POSSIBLE_DEVICE: The best available device
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps:0'
    else:
        device = 'cpu'
    return device
