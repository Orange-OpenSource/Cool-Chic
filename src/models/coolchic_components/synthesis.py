# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


from torch import Tensor, nn
from typing import List, Optional
from models.coolchic_components.base_layers import SynthesisLayer, SynthesisResidualLayer

from models.coolchic_components.quantizable_module import QuantizableModule
from utils.misc import POSSIBLE_Q_STEP_SYN_NN

class Synthesis(QuantizableModule):
    possible_non_linearity = {
        'none': nn.Identity,
        'relu': nn.ReLU,
        'leakyrelu': nn.LeakyReLU,
        'gelu': nn.GELU
    }
    possible_mode = {
        'linear': SynthesisLayer,
        'residual': SynthesisResidualLayer,
    }

    def __init__(self, input_ft: int, layers_dim: List[str]):
        super().__init__(possible_q_steps=POSSIBLE_Q_STEP_SYN_NN)
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for layers in layers_dim:
            out_ft, k_size, mode, non_linearity = layers.split('-')
            out_ft = int(out_ft)
            k_size = int(k_size)

            # Check that mode and non linearity is correct
            assert mode in Synthesis.possible_mode,\
                f'Unknown mode. Found {mode}. Should be in {Synthesis.possible_mode.keys()}'

            assert non_linearity in Synthesis.possible_non_linearity,\
                f'Unknown non linearity. Found {non_linearity}. '\
                f'Should be in {Synthesis.possible_non_linearity.keys()}'

            # Instantiate them
            layers_list.append(
                Synthesis.possible_mode[mode](
                    input_ft,
                    out_ft,
                    k_size,
                    non_linearity=Synthesis.possible_non_linearity[non_linearity](),
                )
            )

            input_ft = out_ft

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
