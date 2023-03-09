# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch

LIST_POSSIBLE_DEVICES = ['cpu', 'cuda:0']

# Avoid numerical instability when measuring the rate of the NN parameters
MIN_SCALE_NN_WEIGHTS_BIAS = 1e-3

# List of all possible scales when coding a MLP
POSSIBLE_SCALE_NN = 10 ** torch.linspace(
    MIN_SCALE_NN_WEIGHTS_BIAS, 1e3, steps=2 ** 16 - 1, device='cpu'
)
# List of all possible quantization steps when coding a MLP
POSSIBLE_Q_STEP_NN = 10. ** torch.linspace(-5, 0, 11, device='cpu')
