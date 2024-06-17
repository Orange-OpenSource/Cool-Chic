# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import numpy as np
import torch
import torch.nn as nn

from enc.utils.misc import FIXED_POINT_FRACTIONAL_MULT, DescriptorNN
from CCLIB.ccencapi import cc_decode_wb


def decode_network(
    empty_module: nn.Module,
    bitstream_path: DescriptorNN,
    q_step_nn: DescriptorNN,
    scale_nn: DescriptorNN,
    ac_max_val: int,
) -> nn.Module:
    """Decode a neural network from a bitstream. The idea is to iterate
    on all the parameters of <empty_module>, filling it with values read
    from the bitstream.

    Args:
        empty_module (nn.Module): An empty (i.e. randomly initialized) instance
            of the network to load.
        bitstream_path (str): Weight and bias will be found at
            <bitstream_path>_weight and <bitstream_path>_arm
        q_step_nn (DescriptorNN): Describe the quantization steps used
            for the weight and bias of the network.
        scale_nn (DescriptorNN): Describe the scale parameters used
            for entropy coding of the weight and bias of the network.
        ac_max_val (int): Data are in [-ac_max_val, ac_max_val - 1]

    Returns:
        nn.Module: The decoded module
    """
    have_bias = q_step_nn.bias > 0

    # Instantiate two range coder objects to decode simultaneously weight and bias
    bac_ctx_weight = cc_decode_wb(bitstream_path.weight)
    if have_bias:
        bac_ctx_bias = cc_decode_wb(bitstream_path.bias)

    loaded_param = {}
    for k, v in empty_module.named_parameters():
        if k.endswith('.w') or k.endswith('.weight'):
            cur_scale = scale_nn.weight
            cur_q_step = q_step_nn.weight
            cur_param = bac_ctx_weight.decode_wb_continue(len(v.flatten()), scale_nn.weight)
        elif k.endswith('.b') or k.endswith('.bias'):
            cur_scale = scale_nn.bias
            cur_q_step = q_step_nn.bias
            cur_param = bac_ctx_bias.decode_wb_continue(len(v.flatten()), scale_nn.bias)
        else:
            # Ignore network parameters whose name does not end with '.w', '.b', '.weight', '.bias'
            continue

        # Don't forget inverse quantization!
        loaded_param[k] = torch.tensor(cur_param).reshape_as(v)  * cur_q_step

    # empty_module.load_state_dict(loaded_param)
    if "arm" in bitstream_path.weight:
        empty_module.set_param_from_float(loaded_param)
    else:
        empty_module.load_state_dict(loaded_param)
    return empty_module
