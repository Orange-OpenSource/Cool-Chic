# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


"""Fixed point implementation of the ARM to avoid floating point drift."""

import math
from dataclasses import fields
from typing import List, Tuple

import torch
from torch import Tensor

from coolchic.bitstream.component.constants import (
    FIXED_POINT_DTYPE,
    LOG_SCALE_MIN_FIXED_POINT,
    MU_MIN_FIXED_POINT,
    N_FRAC_BIT_INTER_FT_CTX,
    WEIGHT_SHIFT,
)
from coolchic.component.core.arm import Arm, ArmLinear
from coolchic.component.core.types import DescriptorNN


def arm_to_fixed_point_param(
    arm: Arm,
    q_steps: DescriptorNN,
    subtract_last_layer: bool = True,
    n_inter_ft_ctx: int = 0,
    no_residual_layer: bool = False,
) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor]:
    """For each layer y = Wx + b, we assume the following accuracy:
        - x: X.8 (X bit of integer part, 8 fractional bits)
        - W: X.8
        - b: X.16
        - y: X.16

    We'll need to shift back y to X.8 after each layer.

    Args:
        arm (Arm): the quantized arm whose parameters will be transformed for fixed
            point arithmetic.
        q_steps (DescriptorNN): Quantization steps used to quantize the weights and biases
        subtract_last_layer (bool, optional): Do we apply the shift_log on the
            biases of the last layer. Typically, there is a -4 shift for the log
            scale. Defaults to True.
        n_inter_ft_ctx (int, optional): How many inter feature contexts are present
            on the first layer input. This is needed because they have a different scaling
            than the latent, being the output of the IFCE. Defaults to 0.
        no_residual_layer (bool, optional): If False, all NxN layers are assumed
            to be residual i.e. y = Wx + b + x. In that case, the residual connexion is factorized
            inside the weight matrix, i.e., y = (W + I)x + b = W'I + b. If no_residual_layer is set
            to True, we don't do that i.e., NxN layers are not assumed to be residual.

    Returns:
        Tuple[List[Tensor], List[Tensor], Tensor, Tensor]: Return 4 elements:
            - A list of fixed point weights for the trunk
            - A list of fixed point biases for the trunk
            - A single fixed point weight tensor for the stabiliser
            - A single fixed point bias tensor for the stabiliser.

        If the ARM has no stabiliser, then the stabiliser weights and biases are set to 0,
        allowing for a branchless forward in the arm.
    """
    fixed_point_weights = []
    fixed_point_biases = []

    fixed_point_param = DescriptorNN(weight=[], bias=[])

    idx_linear_layer = 0
    for i, lay in enumerate(arm.mlp.children()):
        # if isinstance(lay, ArmIntLinear):
        if not isinstance(lay, ArmLinear):
            continue

        is_last_layer = i == (len(list(arm.mlp.children())) - 1)

        for weight_or_bias in fields(DescriptorNN):
            param = lay.__getattr__(weight_or_bias.name)
            if torch.is_floating_point(param) or torch.is_complex(param):
                raise TypeError(f"Quantized parameter should be integer. Found dtype={param.dtype}")

            if weight_or_bias.name == "bias":
                target_shift = WEIGHT_SHIFT * 2
            elif weight_or_bias.name == "weight":
                target_shift = WEIGHT_SHIFT

            # We have already shifted q_param by quantize shift during quantization i.e.
            # q_param = round(param / 2 ** quantize_shift) = round(param << (-quantize_shift))
            quantize_shift = int(math.log2(q_steps.get_value(weight_or_bias.name)))
            actual_shift = target_shift + quantize_shift

            if is_last_layer and weight_or_bias.name == "bias" and subtract_last_layer:
                # There is a hardcoded -4 applied to the log scale on the last layer
                param[1] += -(4 << (-quantize_shift))
                # Requires 64 bits or we explode the dynamics
                # param[1] += -(LOG_SCALE_MIN << (-quantize_shift))
                # param[0] += -(MU_MIN << (-quantize_shift))

            # The inter feature context are already shifted by N_FRAC_BIT_INTER_FT_CTX
            if n_inter_ft_ctx > 0 and weight_or_bias.name == "weight" and i == 0:
                actual_shift = torch.ones_like(param) * actual_shift
                actual_shift[:, -n_inter_ft_ctx:] -= N_FRAC_BIT_INTER_FT_CTX

            fixed_point_param = torch.round(param * (2**actual_shift)).to(FIXED_POINT_DTYPE)

            # All N -> N features layer are residual.
            # Put the residual connexion directly into the weights
            if weight_or_bias.name == "weight":
                if (
                    fixed_point_param.size()[0] == fixed_point_param.size()[1]
                    and not no_residual_layer
                ):
                    identity_matrix = torch.eye(fixed_point_param.size()[0])
                    shift = torch.ones_like(identity_matrix) * target_shift
                    if n_inter_ft_ctx > 0 and weight_or_bias.name == "weight" and i == 0:
                        shift[:, -n_inter_ft_ctx:] -= N_FRAC_BIT_INTER_FT_CTX

                    fixed_point_param += (identity_matrix * (2**shift)).to(FIXED_POINT_DTYPE)

                # Transpose the weights because a linear layer is x @ w.T,
                # that way they are already transposed.
                fixed_point_weights.append(fixed_point_param.T)

            elif weight_or_bias.name == "bias":
                fixed_point_biases.append(fixed_point_param)

        idx_linear_layer += 1

    if arm.flag_linear_stabiliser:
        # For the weights
        target_shift = WEIGHT_SHIFT
        quantize_shift = int(math.log2(q_steps.get_value("weight")))
        actual_shift = target_shift + quantize_shift

        if n_inter_ft_ctx > 0:
            actual_shift = torch.ones_like(arm.stabiliser_branch.weight) * actual_shift
            actual_shift[:, -n_inter_ft_ctx:] -= N_FRAC_BIT_INTER_FT_CTX

        fixed_point_param = torch.round(arm.stabiliser_branch.weight * (2**actual_shift)).to(
            FIXED_POINT_DTYPE
        )
        fixed_point_weight_stabiliser = fixed_point_param.T

        # For the biases
        target_shift = 2 * WEIGHT_SHIFT
        quantize_shift = int(math.log2(q_steps.get_value("bias")))
        actual_shift = target_shift + quantize_shift
        fixed_point_param = torch.round(arm.stabiliser_branch.bias * (2**actual_shift)).to(
            FIXED_POINT_DTYPE
        )
        fixed_point_bias_stabiliser = fixed_point_param

    else:
        fixed_point_weight_stabiliser = torch.zeros(
            (arm.dim_arm, arm.n_out_features), dtype=FIXED_POINT_DTYPE
        )
        fixed_point_bias_stabiliser = torch.zeros((arm.n_out_features), dtype=FIXED_POINT_DTYPE)

    return (
        fixed_point_weights,
        fixed_point_biases,
        fixed_point_weight_stabiliser,
        fixed_point_bias_stabiliser,
    )


MU_LOG_SCALE_MIN_FIXED_POINT = torch.tensor(
    [MU_MIN_FIXED_POINT, LOG_SCALE_MIN_FIXED_POINT], dtype=FIXED_POINT_DTYPE
)


# from line_profiler import profile
# @profile
def fixed_point_arm(
    x: Tensor,
    fixed_point_weights: List[Tensor],
    fixed_point_biases: List[Tensor],
    fixed_point_weights_stab: Tensor,
    fixed_point_biases_stab: Tensor,
    output_shift: int = 0,
) -> Tensor:
    # if torch.is_floating_point(x) or torch.is_complex(x):
    #     raise TypeError(f"Context should be integer. Found dtype={x.dtype}")

    x = x << WEIGHT_SHIFT
    stabiliser = torch.addmm(fixed_point_biases_stab, x, fixed_point_weights_stab)

    # All intermediate layers are residuals --> Already taken into account into w
    for w, b in zip(fixed_point_weights[:-1], fixed_point_biases[:-1]):
        # addmm b, x, w is basically a linear layer with weights w.T and bias b
        # We have already transposed the weight in the arm_to_fixed_point_param
        # We use clamp_min_(0) to mimic a relu as it seems slightly faster
        x = (torch.addmm(b, x, w)).clamp_min_(0) >> WEIGHT_SHIFT

    # Last layer --> no relu
    x = torch.addmm(fixed_point_biases[-1], x, fixed_point_weights[-1]) + stabiliser
    return x >> output_shift
