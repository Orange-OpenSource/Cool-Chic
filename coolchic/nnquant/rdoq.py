# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import time
from dataclasses import fields
from typing import Any, Dict, List, OrderedDict

import torch
from torch import Tensor

from coolchic.component.core.arm import Arm, ArmLinear, Ifce, compute_rate
from coolchic.component.core.coolchic import CoolChicEncoder, exp_golomb_nbins
from coolchic.component.core.synthesis import Synthesis, SynthesisConv2d
from coolchic.component.core.types import DescriptorNN
from coolchic.component.core.upsampling import fixed_upsampling
from coolchic.component.frame import FrameEncoder
from coolchic.io.format.yuv import convert_444_to_420, yuv_dict_clamp
from coolchic.training.loss import DISTORTION_METRIC, loss_function
from coolchic.training.test import test
from coolchic.utils.codingstructure import Frame


@torch.no_grad()
def rdoq_model(
    frame_encoder: FrameEncoder,
    frame: Frame,
    dist_weight: Dict[DISTORTION_METRIC, float],
    lmbda: float,
    verbosity: int = 0,
) -> FrameEncoder:
    """Adjust the discrete value of the quantized neural networks with a greedy algorithm.

    Each neural network parameters is expressed as k * q_step with k an integer. This function
    tries out different value of k +/- 1, 2, 3 etc. to refine the quantized NNs.

    Args:
        frame_encoder: Quantized frame encoder to be refined.
        frame: Original (non compressed) target frame.
        dist_weight: Dictionnary containing the weighting of all distortion metrics e.g., {"mse": 1.0}
        lmbda: Rate constraint
        verbosity: Verbosity level. Defaults to 0.

    Returns:
        FrameEncoder: Refined frame encoder, ready to be written to a bitstream
    """
    start_time = time.time()

    print("Start discrete RDOQ on neural network parameters...")
    print("Initial performance:")

    encoder_logs = test(frame_encoder, frame, dist_weight, lmbda)
    print(encoder_logs.pretty_string(show_col_name=True, mode="short") + "\n")

    test_cnt = 0

    for cc_name, cc_enc in frame_encoder.coolchic_enc.items():
        # Quantize all the modules we send
        module_to_rdoq = {
            module_name: getattr(cc_enc, module_name) for module_name in cc_enc.modules_to_send
        }
        print_col_name_summary = True

        for module_name, cur_module in sorted(module_to_rdoq.items()):
            # Loop on weights and biases
            for w_b in fields(DescriptorNN):
                print_col_name_verbose = True
                weight_or_bias = w_b.name

                param = cur_module.get_param(which=weight_or_bias)
                flat_param = torch.cat([v.flatten() for _, v in param.items()])
                dict_size = {k: torch.tensor(v.size()) for k, v in param.items()}

                # Partial forward for some modules to go faster
                if module_name == "arm":
                    flat_latent, flat_context = cc_enc.get_latent_context(
                        cc_enc.get_quantize_latent(
                            quantizer_noise_type="none", quantizer_type="hardround"
                        )
                    )
                    best_loss = score_arm(
                        cur_module,
                        flat_latent,
                        flat_context,
                        cc_enc.nn_q_step.get_value(module_name),
                        cc_enc.nn_expgol_cnt.get_value(module_name),
                    )

                elif module_name == "ifce":
                    decoder_side_latent = cc_enc.get_quantize_latent(
                        quantizer_noise_type="none", quantizer_type="hardround"
                    )
                    _, intermediate_latent_ups = fixed_upsampling(
                        decoder_side_latent, mode="nearest"
                    )
                    spatial_ctx, flat_latent = cc_enc.get_spatial_context_flat_latent(
                        decoder_side_latent
                    )

                    best_loss = score_ifce(
                        cur_module,
                        cc_enc,
                        decoder_side_latent,
                        intermediate_latent_ups,
                        flat_latent,
                        spatial_ctx,
                        cc_enc.nn_q_step.get_value(module_name),
                        cc_enc.nn_expgol_cnt.get_value(module_name),
                    )

                elif module_name == "synthesis" and frame.frame_type == "I":
                    latent = cc_enc.get_quantize_latent(
                        quantizer_noise_type="none", quantizer_type="hardround"
                    )
                    latent = cc_enc.discard_hyperlatent(latent)
                    syn_in = cc_enc.get_synthesis_input(cc_enc.upsampling(latent))

                    best_loss = score_syn_image(
                        cur_module,
                        cc_enc,
                        syn_in,
                        frame,
                        dist_weight,
                        lmbda,
                        cc_enc.nn_q_step.get_value(module_name),
                        cc_enc.nn_expgol_cnt.get_value(module_name),
                    )

                # Usual complete forward
                else:
                    encoder_logs = test(frame_encoder, frame, dist_weight, lmbda)
                    best_loss = encoder_logs.loss
                # print(encoder_logs.pretty_string(show_col_name=True, mode="short"))

                q_step = cc_enc.nn_q_step.get_value(module_name, weight_or_bias)
                random_idx = torch.randperm(flat_param.numel())

                for j, idx_val in enumerate(random_idx):
                    initial_value = float(flat_param[idx_val])
                    best_value = initial_value

                    for sign in [-1, 1]:
                        for shift in range(1, 16):
                            flat_param[idx_val] = initial_value + sign * shift * q_step

                            getattr(cc_enc, module_name).set_param(
                                flat_to_2d(dict_size, flat_param), strict=False
                            )

                            test_cnt += 1

                            if module_name == "arm":
                                candidate_loss = score_arm(
                                    cur_module,
                                    flat_latent,
                                    flat_context,
                                    cc_enc.nn_q_step.get_value(module_name),
                                    cc_enc.nn_expgol_cnt.get_value(module_name),
                                )
                            elif module_name == "ifce":
                                candidate_loss = score_ifce(
                                    cur_module,
                                    cc_enc,
                                    decoder_side_latent,
                                    intermediate_latent_ups,
                                    flat_latent,
                                    spatial_ctx,
                                    cc_enc.nn_q_step.get_value(module_name),
                                    cc_enc.nn_expgol_cnt.get_value(module_name),
                                )
                            elif module_name == "synthesis" and frame.frame_type == "I":
                                candidate_loss = score_syn_image(
                                    cur_module,
                                    cc_enc,
                                    syn_in,
                                    frame,
                                    dist_weight,
                                    lmbda,
                                    cc_enc.nn_q_step.get_value(module_name),
                                    cc_enc.nn_expgol_cnt.get_value(module_name),
                                )
                            else:
                                encoder_logs = test(frame_encoder, frame, dist_weight, lmbda)
                                candidate_loss = encoder_logs.loss

                            if candidate_loss < best_loss:
                                best_loss = candidate_loss
                                best_value = float(flat_param[idx_val])

                                if verbosity > 0:
                                    print(
                                        encoder_logs.pretty_string(
                                            show_col_name=print_col_name_verbose,
                                            mode="short",
                                            additional_data={
                                                "shift": f"{shift * sign:>4}",
                                                "cc_name": cc_name,
                                                "module": module_name,
                                                "w_or_b": weight_or_bias,
                                                "rdoq_time": f"{time.time() - start_time:7.1f}",
                                                "n_forward": test_cnt,
                                                "cnt": j,
                                            },
                                        )
                                    )
                                print_col_name_verbose = False
                            # Early stopping to avoid multiple forwards
                            else:
                                break

                    flat_param[idx_val] = best_value

                    cur_module.set_param(flat_to_2d(dict_size, flat_param), strict=False)
                    param = cur_module.get_param(which=weight_or_bias)
                    flat_param = torch.cat([v.flatten() for _, v in param.items()])

                encoder_logs = test(frame_encoder, frame, dist_weight, lmbda)

                print(
                    encoder_logs.pretty_string(
                        show_col_name=print_col_name_summary,
                        mode="short",
                        additional_data={
                            "cc_name": cc_name,
                            "module": module_name,
                            "w_or_b": weight_or_bias,
                            "rdoq_time": f"{time.time() - start_time:7.1f}",
                            "n_forward": test_cnt,
                        },
                    )
                )
                print_col_name_summary = False

    time_nn_rdoq = time.time() - start_time

    print(f"\nTime rdoq_model(): {time_nn_rdoq:4.1f} seconds\n")
    frame_encoder.encoder_monitor.increment_time(time_nn_rdoq)

    return frame_encoder


def flat_to_2d(
    dict_size: OrderedDict[str, torch.Size], flat_param: Tensor
) -> OrderedDict[str, Tensor]:
    """Transform a 1-dimensional tensor into a dictionary of parameters whose
    individual size is described in dict_size.

    Args:
        dict_size: Size of all parameters.
        flat_param (Tensor): Concatenation of the parameters in a single flat tensor.

    Returns:
        OrderedDict[str, Tensor]: Dictionary of parameters with the proper size.
    """
    param = {}
    ptr = 0
    for k, v in dict_size.items():
        numel = torch.cumprod(v, dim=0)[-1]
        param[k] = flat_param[ptr : ptr + numel].view(v.tolist())
        ptr += numel
    return param


def _measure_latent_rate(arm: Arm, flat_latent: Tensor, flat_context: Tensor) -> Tensor:
    """Measure the rate in bits of latents, given the ARM and its context.

    Args:
        arm: ARM module to predict the latent distribution.
        flat_latent: 1-dimensional tensor gathering the latent. Shape: [B]
        flat_context: C-dimensional context for each latent. Shape [B, C]

    Returns:
        Tensor: Total rate in bits.
    """
    flat_mu, flat_scale = arm.reparameterize_output(arm(flat_context))
    rate = compute_rate(flat_latent, flat_mu, flat_scale)
    return rate.sum()


def _measure_nn_rate(
    layer_list: List[Any], q_step: DescriptorNN, expgol_cnt: DescriptorNN
) -> float:
    """Measure the rate of the NN parameters based on a list of layer.

    Args:
        layer_list: List of neural network layers whose rate is measured.
        q_step: Quantization steps for the biases and weights.
        expgol_cnt: Exp-golomb order for the baises and weights

    Returns:
        float: Total number of bits for all the layers.
    """

    rate = 0
    for layer in layer_list:
        sent_w = torch.round(layer.weight / q_step.get_value("weight"))
        sent_b = torch.round(layer.bias / q_step.get_value("bias"))

        bits_weights = exp_golomb_nbins(sent_w, expgol_cnt.get_value("weight"))
        bits_biases = exp_golomb_nbins(sent_b, expgol_cnt.get_value("bias"))

        rate += bits_weights + bits_biases

    return rate


# ---
# --- Partial forwards for faster evaluations
# ---


@torch.no_grad()
def score_arm(
    arm: Arm,
    flat_latent: Tensor,
    flat_context: Tensor,
    q_step: DescriptorNN,
    expgol_cnt: DescriptorNN,
) -> Tensor:
    """Test an ARM by measuring the rate of the latent and the rate of NN parameters.

    Args:
        arm: ARM module to predict the latent distribution.
        flat_latent: 1-dimensional tensor gathering the latent. Shape: [B]
        flat_context: C-dimensional context for each latent. Shape [B, C]
        q_step: Quantization steps for the biases and weights.
        expgol_cnt: Exp-golomb order for the baises and weights

    Returns:
        Tensor: Total rate in bits of the latents and the ARM parameters.
    """

    layer_list = [layer for layer in arm.mlp if isinstance(layer, ArmLinear)]

    if arm.flag_linear_stabiliser:
        layer_list += [arm.stabiliser_branch]

    rate_latent = _measure_latent_rate(arm, flat_latent, flat_context)
    rate_nn = _measure_nn_rate(layer_list, q_step, expgol_cnt)
    return rate_latent + rate_nn


@torch.no_grad()
def score_ifce(
    ifce: Ifce,
    cc_enc: CoolChicEncoder,
    decoder_side_latent: List[Tensor],
    intermediate_latent_ups: List[Tensor],
    flat_latent: Tensor,
    spatial_ctx: Tensor,
    q_step: DescriptorNN,
    expgol_cnt: DescriptorNN,
) -> Tensor:
    """Test an IFCE by measuring the rate of the latent and the rate of NN parameters.

    Args:
        ifce: IFCE module to predict the latent distribution.
        cc_enc: Cool-chic encoder, required to access the ARM used to interpret the IFCE output.
        decoder_side_latent: List of quantized latent grids
        intermediate_latent_ups: List of the intermediate upsampling used by the IFCE to extracty
            co-localized contexts.
        flat_latent: 1-dimensional tensor gathering the latent. Shape: [B]
        spatial_ctx: S-dimensional spatial context for each latent. Shape [B, S]
        q_step: Quantization steps for the biases and weights.
        expgol_cnt: Exp-golomb order for the baises and weights

    Returns:
        Tensor: Total rate in bits of the latents and the IFCE parameters.
    """

    layer_list = []
    for arm in ifce.arms:
        layer_list += [layer for layer in arm.mlp if isinstance(layer, ArmLinear)]

    ifce_ctx = cc_enc.get_ifce_output(decoder_side_latent, intermediate_latent_ups)
    flat_context = torch.cat((spatial_ctx, ifce_ctx), dim=1)

    rate_latent = _measure_latent_rate(cc_enc.arm, flat_latent, flat_context)
    rate_nn = _measure_nn_rate(layer_list, q_step, expgol_cnt)
    return rate_latent + rate_nn


@torch.no_grad()
def score_syn_image(
    synthesis: Synthesis,
    cc_enc: CoolChicEncoder,
    syn_in: Tensor,
    frame: Frame,
    dist_weight: Dict[DISTORTION_METRIC, float],
    lmbda: float,
    q_step: DescriptorNN,
    expgol_cnt: DescriptorNN,
):
    """Test the I-frame synthesis by returning the following score:

        score = dist(x, \\hat{x}) + lmbda * Rate(synthesis) parameters.

    Since the latent rate is constant, it is ignored in the cost above.

    Args:
        synthesis: The synthesis to score
        cc_enc: Cool-chic encoder, required to use the final upsampling after the synthesis.
        syn_in: Synthesis input. Shape is [1, C, H, W]
        frame: Original (non compressed) target frame.
        dist_weight: Dictionnary containing the weighting of all distortion metrics e.g., {"mse": 1.0}
        lmbda: Rate constraint
        q_step: Quantization steps for the biases and weights.
        expgol_cnt: Exp-golomb order for the baises and weights

    Returns:
        Tensor: Synthesis score
    """

    # Partial forward, only I-frame.
    if frame.frame_type != "I":
        raise ValueError(
            f"score_syn_image can only be used for I-frame. Found frame_type={frame.frame_type}"
        )

    # ------ Reconstruct the image from the latent
    decoded_image = cc_enc.rescale_output(synthesis(syn_in))
    # Downsample if necessary and constraint range
    if frame.data.frame_data_type == "yuv420":
        decoded_image = yuv_dict_clamp(convert_444_to_420(decoded_image), 0.0, 1.0)
    elif frame.data.frame_data_type != "flow":
        decoded_image = torch.clamp(decoded_image, 0.0, 1.0)

    # Decoded frame quantization
    max_dynamic = 2 ** (frame.data.bitdepth) - 1
    if frame.data.frame_data_type == "yuv420":
        for k, v in decoded_image.items():
            decoded_image[k] = torch.round(v * max_dynamic) / max_dynamic
    else:
        decoded_image = torch.round(decoded_image * max_dynamic) / max_dynamic

    # -------- Get the neural network rate
    layer_list = [
        layer for layer in synthesis.main_branch if isinstance(layer, SynthesisConv2d)
    ] + [synthesis.output_transform]

    if synthesis.flag_linear_stabiliser:
        layer_list += [synthesis.stabiliser_branch]
    rate_nn = _measure_nn_rate(layer_list, q_step, expgol_cnt)

    # -------- We can ignore the latent rate here
    rate_latent = {"residue": torch.zeros((1,), device=syn_in.device)}

    # ------- Call the loss
    score = loss_function(
        decoded_image,
        rate_latent,
        frame.data.data,
        dist_weight,
        lmbda=lmbda,
        total_rate_nn_bit=rate_nn,
        compute_logs=False,
    ).loss
    return score
