# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from enc.component.types import DescriptorNN
from enc.nnquant.quantstep import get_q_step_from_parameter_name
import torch
from torch import Tensor, nn


POSSIBLE_EXP_GOL_COUNT = {
    "arm": {
        "weight": torch.linspace(0, 12, 13, device="cpu"),
        "bias": torch.linspace(0, 12, 13, device="cpu"),
    },
    "upsampling": {
        "weight": torch.linspace(0, 12, 13, device="cpu"),
        "bias": torch.linspace(0, 12, 13, device="cpu"),
    },
    "synthesis": {
        "weight": torch.linspace(0, 12, 13, device="cpu"),
        "bias": torch.linspace(0, 12, 13, device="cpu"),
    },
}


@torch.no_grad()
def measure_expgolomb_rate(
    q_module: nn.Module, q_step: DescriptorNN, expgol_cnt: DescriptorNN
) -> DescriptorNN:
    """Get the rate associated with the current parameters.

    Returns:
        DescriptorNN: The rate of the different modules wrapped inside a dictionary
            of float. It does **not** return tensor so no back propagation is possible
    """
    # Concatenate the sent parameters here to measure the entropy later
    sent_param: DescriptorNN = {"bias": [], "weight": []}
    rate_param: DescriptorNN = {"bias": 0.0, "weight": 0.0}

    param = q_module.get_param()
    # Retrieve all the sent item
    for parameter_name, parameter_value in param.items():
        current_q_step = get_q_step_from_parameter_name(parameter_name, q_step)
        # Current quantization step is None because the module is not yet
        # quantized. Return an all zero rate
        if current_q_step is None:
            return rate_param

        # Quantization is round(parameter_value / q_step) * q_step so we divide by q_step
        # to obtain the sent latent.
        current_sent_param = (parameter_value / current_q_step).view(-1)

        if ".weight" in parameter_name:
            sent_param["weight"].append(current_sent_param)
        elif ".bias" in parameter_name:
            sent_param["bias"].append(current_sent_param)
        else:
            print(
                'Parameter name should include ".weight" or ".bias" '
                f"Found: {parameter_name}"
            )
            return rate_param

    # For each sent parameters (e.g. all biases and all weights)
    # compute their cost with an exp-golomb coding.
    for k, v in sent_param.items():
        # If we do not have any parameter, there is no rate associated.
        # This can happens for the upsampling biases for instance
        if len(v) == 0:
            rate_param[k] = 0.0
            continue

        # Current exp-golomb count is None because the module is not yet
        # quantized. Return an all zero rate
        current_expgol_cnt = expgol_cnt[k]
        if current_expgol_cnt is None:
            return rate_param

        # Concatenate the list of parameters as a big one dimensional tensor
        v = torch.cat(v)

        # This will be pretty long! Could it be vectorized?
        rate_param[k] = exp_golomb_nbins(v, count=current_expgol_cnt)

    return rate_param


def exp_golomb_nbins(symbol: Tensor, count: int = 0) -> Tensor:
    """Compute the number of bits required to encode a Tensor of integers
    using an exponential-golomb code with exponent ``count``.

    Args:
        symbol: Tensor to encode
        count (int, optional): Exponent of the exp-golomb code. Defaults to 0.

    Returns:
        Number of bits required to encode all the symbols.
    """

    # We encode the sign equiprobably at the end thus one more bit if symbol != 0
    nbins = (
        2 * torch.floor(torch.log2(symbol.abs() / (2**count) + 1))
        + count
        + 1
        + (symbol != 0)
    )
    res = nbins.sum()
    return res
