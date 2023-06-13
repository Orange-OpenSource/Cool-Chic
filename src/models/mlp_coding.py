# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


"""Gather function used to quantize and entropy code an MLP."""

import itertools
import sys
import time
import torch

from model_management.loss import loss_fn
from models.cool_chic import CoolChicEncoder, to_device
from utils.constants import FIXED_POINT_FRACTIONAL_MULT, ARMINT
from utils.data_structure import DescriptorNN


@torch.no_grad()
def greedy_quantization(model: CoolChicEncoder) -> CoolChicEncoder:
    """Return a quantized model.
    # ! We also obtain the integerized ARM here!

    Args:
        model (CoolChicEncoder): Model to be quantized

    Returns:
        CoolChicEncoder: The quantized model
    """

    start_time = time.time()

    module_to_quantize = {
        'arm': model.arm, 'upsampling': model.upsampling, 'synthesis': model.synthesis,
    }

    best_q_step = {k: None for k in module_to_quantize}

    for module_name, module in module_to_quantize.items():
        # Start the RD optimization for the quantization step of each module with an
        # arbitrary high value for the RD cost.
        best_loss = 1e6

        # Save full precision parameters before quantizing
        module.save_full_precision_param()

        # Try to find the best quantization step
        all_q_step = module._POSSIBLE_Q_STEP
        for q_step_w, q_step_b in itertools.product(all_q_step, all_q_step):
            # Quantize
            current_q_step: DescriptorNN = {'weight': q_step_w, 'bias': q_step_b}
            quantization_success = module.quantize(current_q_step)
            if not quantization_success:
                continue

            # Measure rate
            rate_per_module = module.measure_laplace_rate()
            total_rate_module_bit = sum([v for _, v in rate_per_module.items()])

            # Evaluate
            model = model.eval()

            # From Gordon's code
            if module_name == "arm":
                if ARMINT:
                    model = to_device(model, 'cpu')
                model.arm.set_quant(FIXED_POINT_FRACTIONAL_MULT)

            model_out = model()
            # Compute results
            loss, _ = loss_fn(
                model_out,
                model.param.img,
                model.param.lmbda,
                compute_logs=False,
                dist_mode=model.param.dist,
                rate_mlp=total_rate_module_bit
            )

            # Store best quantization steps
            if loss < best_loss:
                best_loss = loss
                best_q_step[module_name] = current_q_step

        # Once we've tested all the possible quantization step: quantize one last
        # time with the best one we've found to actually use it.
        quantization_success = module.quantize(best_q_step[module_name])
        if not quantization_success:
            print(f'Greedy quantization failed!')
            sys.exit(0)

    print(f'\nTime greedy_quantization: {time.time() - start_time:4.1f} seconds\n')

    # Re-apply integerization of the module
    model.arm.set_quant(FIXED_POINT_FRACTIONAL_MULT)

    return model
