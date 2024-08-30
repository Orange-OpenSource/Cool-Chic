# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import itertools
import time
from typing import Optional, OrderedDict

import torch
from enc.utils.misc import exp_golomb_nbins
from enc.training.loss import loss_function
from enc.utils.manager import FrameEncoderManager
from enc.component.frame import FrameEncoder
from enc.utils.codingstructure import Frame
from enc.utils.misc import (
    MAX_AC_MAX_VAL,
    POSSIBLE_EXP_GOL_COUNT,
    POSSIBLE_Q_STEP,
    DescriptorNN,
    get_q_step_from_parameter_name,
)
from torch import Tensor


def _quantize_parameters(
    fp_param: OrderedDict[str, Tensor],
    q_step: DescriptorNN,
) -> Optional[OrderedDict[str, Tensor]]:
    """Quantize a dictionary of parameters fp_param with a given quantization
    step (e.g. one for bias one for the weight).
    Return None if quantization fails i.e. if round(param / q_step) is greater
    than MAC_AX_MAX_VAL.

    Args:
        fp_param (OrderedDict[str, Tensor]): Full precision parameter, usually
            the output of self.get_param() or self.named_parameters()
        q_step (DescriptorNN): A dictionary with one quantization step for the
            weight and one for the bias.

    Returns:
        Optional[OrderedDict[str, Tensor]]: The quantized parameters or None
            if quantization failed.
    """
    q_param = OrderedDict()
    for k, v in fp_param.items():
        current_q_step = get_q_step_from_parameter_name(k, q_step)
        sent_param = torch.round(v / current_q_step)

        if sent_param.abs().max() > MAX_AC_MAX_VAL:
            print(
                f"Sent param {k} exceed MAX_AC_MAX_VAL! Q step {current_q_step} too small."
            )
            return None

        q_param[k] = sent_param * current_q_step

    return q_param

@torch.no_grad()
def quantize_model(
    frame_encoder: FrameEncoder,
    frame: Frame,
    frame_encoder_manager: FrameEncoderManager,
) -> FrameEncoder:
    """Quantize a ``FrameEncoder`` compressing a ``Frame`` under a rate
    constraint ``lmbda`` and return it.

    This function iterates on all the neural networks sent from the encoder
    to the decoder, listed in `frame_encoder.coolchic_encoder.modules_to_send`.
    For each module :math:`m`, we want to find the most suited pair of
    quantization steps for the weight and the biases :math:`(\\Delta_w^m,
    \\Delta_b^m)`.

    To do so, a greedy search is used where we quantize the weights and biases
    using all the possible pairs of quantization steps, and we compute the
    :doc`usual loss function <./loss>`. The loss measures the impact of the NN
    quantization steps :math:`(\\Delta_w^m, \\Delta_b^m)` on the MSE / rate of
    the decoded image and the rate of the NN.-

    In the end, we select the pair of quantization step minimizing the loss:

        .. math::

            (\\Delta_w^m, \\Delta_b^m) = \\arg\\min ||\\mathbf{x}
            - \hat{\\mathbf{x}}||^2 + \\lambda
            (\\mathrm{R}(\hat{\\mathbf{x}}) + \\mathrm{R}_{NN}), \\text{ with }
            \\begin{cases}
                \\mathbf{x} & \\text{the original image}\\\\ \\hat{\\mathbf{x}} &
                \\text{the coded image}\\\\ \\mathrm{R}(\\hat{\\mathbf{x}}) &
                \\text{A measure of the rate of } \\hat{\\mathbf{x}} \\\\
                    \\mathrm{R}_{NN} & \\text{The rate of the neural networks}
            \\end{cases}

    Then we quantize the next module to be sent.

    .. warning::

        The parameter ``frame_encoder_manager`` tracking the encoding time of
        the frame (``total_training_time_sec``) and the number of encoding
        iterations (``iterations_counter``) is modified ** in place** by this
        function.


    Args:
        frame_encoder: Model to be compressed.
        frame: Original frame to code, including its references.
        frame_encoder_manager: Contains (among other things) the rate
            constraint :math:`\\lambda` and description of the warm-up preset.
            It is also used to track the total encoding time and encoding
            iterations. Modified in place.

    Returns:
        Model with quantized parameters.
    """
    start_time = time.time()
    frame_encoder.set_to_eval()

    # We have to quantize all the modules that we want to send
    module_to_quantize = {
        module_name: getattr(frame_encoder.coolchic_encoder, module_name)
        for module_name in frame_encoder.coolchic_encoder.modules_to_send
    }

    for module_name, cur_module in sorted(module_to_quantize.items()):
        # Start the RD optimization for the quantization step of each module with an
        # arbitrary high value for the RD cost.
        best_loss = 1e6

        # All possible quantization steps for this module
        all_q_step = POSSIBLE_Q_STEP.get(module_name)
        all_expgol_cnt = POSSIBLE_EXP_GOL_COUNT.get(module_name)

        # Save full precision parameter.
        fp_param = cur_module.get_param()

        best_q_step = {}
        # Overall best expgol count for this module weights and biases
        final_best_expgol_cnt = {}

        for q_step_w, q_step_b in itertools.product(all_q_step.get("weight"), all_q_step.get("bias")):
            # Reset full precision parameters, set the quantization step
            # and quantize the model.
            current_q_step: DescriptorNN = {"weight": q_step_w, "bias": q_step_b}

            # Reset full precision parameter before quantizing
            q_param = _quantize_parameters(fp_param, current_q_step)

            # Quantization has failed
            if q_param is None:
                continue

            cur_module.set_param(q_param)

            # Plug the quantized module back into Cool-chic
            setattr(frame_encoder.coolchic_encoder, module_name, cur_module)

            frame_encoder.coolchic_encoder.nn_q_step[module_name] = current_q_step

            # Test Cool-chic performance with this quantization steps pair
            frame_encoder_out = frame_encoder.forward(
                reference_frames=[ref_i.data for ref_i in frame.refs_data],
                quantizer_noise_type="none",
                quantizer_type="hardround",
                AC_MAX_VAL=-1,
                flag_additional_outputs=False,
            )

            param = cur_module.get_param()

            # Best exp-golomb count for this quantization step
            best_expgol_cnt = {}
            for weight_or_bias in ["weight", "bias"]:

                # Find the best exp-golomb count for this quantization step:
                cur_best_expgol_cnt = None
                # Arbitrarily high number
                cur_best_rate = 1e9

                sent_param = []
                for parameter_name, parameter_value in param.items():

                    # Quantization is round(parameter_value / q_step) * q_step so we divide by q_step
                    # to obtain the sent latent.
                    current_sent_param = (parameter_value / current_q_step.get(weight_or_bias)).view(-1)

                    if parameter_name.endswith(weight_or_bias):
                        sent_param.append(current_sent_param)

                # Integer, sent parameters
                v = torch.cat(sent_param)

                # Find the best expgol count for this weight
                for expgol_cnt in all_expgol_cnt.get(weight_or_bias):
                    cur_rate = exp_golomb_nbins(v, count=expgol_cnt)
                    if cur_rate < cur_best_rate:
                        cur_best_rate = cur_rate
                        cur_best_expgol_cnt = expgol_cnt

                best_expgol_cnt[weight_or_bias] = int(cur_best_expgol_cnt)

            frame_encoder.coolchic_encoder.nn_expgol_cnt[module_name] = best_expgol_cnt

            rate_mlp = 0.0
            rate_per_module = frame_encoder.coolchic_encoder.get_network_rate()
            for _, module_rate in rate_per_module.items():
                for _, param_rate in module_rate.items():  # weight, bias
                    rate_mlp += param_rate

            loss_fn_output = loss_function(
                frame_encoder_out.decoded_image,
                frame_encoder_out.rate,
                frame.data.data,
                lmbda=frame_encoder_manager.lmbda,
                rate_mlp_bit=rate_mlp,
                compute_logs=True,
            )


            # Store best quantization steps
            if loss_fn_output.loss < best_loss:
                best_loss = loss_fn_output.loss
                best_q_step = current_q_step
                final_best_expgol_cnt = best_expgol_cnt

        # Once we've tested all the possible quantization step and expgol_cnt,
        # quantize one last time with the best one we've found to actually use it.
        frame_encoder.coolchic_encoder.nn_q_step[module_name] = best_q_step
        frame_encoder.coolchic_encoder.nn_expgol_cnt[module_name] = final_best_expgol_cnt

        q_param = _quantize_parameters(fp_param, frame_encoder.coolchic_encoder.nn_q_step[module_name])
        assert q_param is not None, (
            "_quantize_parameters() failed with q_step "
            f"{frame_encoder.coolchic_encoder.nn_q_step[module_name]}"
        )

        cur_module.set_param(q_param)
        # Plug the quantized module back into Cool-chic
        setattr(frame_encoder.coolchic_encoder, module_name, cur_module)

    time_nn_quantization = time.time() - start_time

    print(f"\nTime quantize_model(): {time_nn_quantization:4.1f} seconds\n")
    frame_encoder_manager.total_training_time_sec += time_nn_quantization

    return frame_encoder
