# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


"""All possible quantization steps for each module of a CoolchicEncoder"""

from typing import Optional

from enc.component.types import DescriptorNN
import torch

# Shifts for ARM, record the shift.
POSSIBLE_Q_STEP_SHIFT = {
    "arm": {
        "weight": torch.linspace(-8, 0, 9, device="cpu"),
        "bias": torch.linspace(-16, 0, 17, device="cpu"),
    },
}

POSSIBLE_Q_STEP = {
    "arm": {
        "weight": 2.0 ** POSSIBLE_Q_STEP_SHIFT["arm"]["weight"],
        "bias": 2.0 ** POSSIBLE_Q_STEP_SHIFT["arm"]["bias"],
    },
    "upsampling": {
        "weight": 2.0 ** torch.linspace(-12, 0, 13, device="cpu"),
        "bias": 2.0 ** torch.tensor([0.0]),
    },
    "synthesis": {
        "weight": 2.0 ** torch.linspace(-12, 0, 13, device="cpu"),
        "bias": 2.0 ** torch.linspace(-24, 0, 25, device="cpu"),
    },
}


def get_q_step_from_parameter_name(
    parameter_name: str, q_step: DescriptorNN
) -> Optional[float]:
    """Return the specific quantization step from q_step (a dictionary
    with several quantization steps). The specific quantization step is
    selected through the parameter name.

    Args:
        parameter_name (str): Name of the parameter in the state dict.
        q_step (DescriptorNN): Dictionary gathering several quantization
            steps. E.g. one quantization step for the weights and one for
            the biases.

    Returns:
        Optional[float]: The quantization step associated to the parameter.
            Return None if nothing is found.
    """
    if ".weight" in parameter_name:
        current_q_step = q_step.get("weight")
    elif ".bias" in parameter_name:
        current_q_step = q_step.get("bias")
    else:
        print(
            'Parameter name should include ".weight" or ".bias" '
            f"Found: {parameter_name}"
        )
        current_q_step = None

    return current_q_step
