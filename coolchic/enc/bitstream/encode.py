# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import math
import os
import subprocess
import time

import torch

from dec.nn import decode_network

from enc.bitstream.utils import get_sub_bitstream_path
from enc.bitstream.header import write_frame_header, write_gop_header
from CCLIB.ccencapi import cc_code_latent_layer_bac, cc_code_wb_bac
from enc.component.core.arm import Arm, ArmInt
from enc.component.core.synthesis import Synthesis
from enc.component.core.upsampling import Upsampling
from enc.component.frame import FrameEncoder
from enc.component.video import VideoEncoder
from enc.utils.misc import (
    FIXED_POINT_FRACTIONAL_MULT,
    FIXED_POINT_FRACTIONAL_BITS,
    POSSIBLE_Q_STEP_SHIFT,
    POSSIBLE_Q_STEP,
    DescriptorCoolChic,
    DescriptorNN,
)


def get_ac_max_val_nn(frame_encoder: FrameEncoder) -> int:
    """Look within the neural networks of a frame encoder. Return the maximum
    amplitude of the quantized model (i.e. weight / q_step).
    This allows to get the AC_MAX_VAL constant, used by the entropy coder. All
    symbols sent by the entropy coder will be in [-AC_MAX_VAL, AC_MAX_VAL - 1].

    Args:
        frame_encoder (FrameEncoder): Model to quantize.

    Returns:
        int: The AC_MAX_VAL parameter.
    """
    model_param_quant = []

    # Loop on all the modules to be sent, and find the biggest quantized value
    for cur_module_name in frame_encoder.coolchic_encoder.modules_to_send:
        module_to_encode = getattr(frame_encoder.coolchic_encoder, cur_module_name)

        # Retrieve all the weights and biases for the ARM MLP
        for k, v in module_to_encode.named_parameters():

            if k.endswith('.w') or k.endswith('.weight'):
                cur_possible_q_step = POSSIBLE_Q_STEP.get(cur_module_name).get("weight")

                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step = frame_encoder.coolchic_encoder.nn_q_step.get(
                    cur_module_name
                ).get("weight")
                cur_q_step_index = int(
                    torch.argmin((cur_possible_q_step - cur_q_step).abs()).item()
                )

                # Quantize the weight with the actual quantization step and add it
                # to the list of (quantized) weights
                if cur_module_name == "arm":
                    # to float, then qstep
                    model_param_quant.append(
                        torch.round((v/FIXED_POINT_FRACTIONAL_MULT) / cur_possible_q_step[cur_q_step_index]).flatten()
                    )
                else:
                    model_param_quant.append(
                        torch.round(v / cur_possible_q_step[cur_q_step_index]).flatten()
                    )


            elif k.endswith('.b') or k.endswith('.bias'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_possible_q_step = POSSIBLE_Q_STEP.get(cur_module_name).get("bias")
                cur_q_step = frame_encoder.coolchic_encoder.nn_q_step.get(
                    cur_module_name
                ).get("bias")
                cur_q_step_index = int(
                    torch.argmin((cur_possible_q_step - cur_q_step).abs()).item()
                )
                # Quantize the bias with the actual quantization step and add it
                # to the list of (quantized) bias
                if cur_module_name == "arm":
                    # to float, then qstep
                    model_param_quant.append(
                        torch.round((v/FIXED_POINT_FRACTIONAL_MULT/FIXED_POINT_FRACTIONAL_MULT) / cur_possible_q_step[cur_q_step_index]).flatten()
                    )
                else:
                    model_param_quant.append(
                        torch.round(v / cur_possible_q_step[cur_q_step_index]).flatten()
                    )

    # Gather them
    model_param_quant = torch.cat(model_param_quant).flatten()

    # Compute AC_MAX_VAL.
    AC_MAX_VAL = int(torch.ceil(model_param_quant.abs().max() + 2).item())
    return AC_MAX_VAL


def get_ac_max_val_latent(frame_encoder: FrameEncoder) -> int:
    """Look within the latent variables of a frame encoder. Return the maximum
    amplitude of the quantized latent variables.
    This allows to get the AC_MAX_VAL constant, used by the entropy coder. All
    symbols sent by the entropy coder will be in [-AC_MAX_VAL, AC_MAX_VAL - 1].

    Args:
        frame_encoder (FrameEncoder): Model storing the latent.

    Returns:
        int: The AC_MAX_VAL parameter.
    """
    # Setting flag_additional_outputs=True allows to recover the quantized latent.
    # Don't specify AC_MAX_VAL now: we let the latents evolve freely to capture
    # their dynamic.
    encoder_output = frame_encoder.coolchic_encoder.forward(
        quantizer_noise_type="none",
        quantizer_type="hardround",
        AC_MAX_VAL=-1,
        flag_additional_outputs=True,
    )
    latent = torch.cat(
        [
            y_i.flatten()
            for y_i in encoder_output.get("additional_data").get("detailed_sent_latent")
        ],
        dim=0,
    )

    # Compute AC_MAC_VAL
    AC_MAX_VAL = int(torch.ceil(latent.abs().max() + 2).item())
    return AC_MAX_VAL


def encode_video(video_encoder: VideoEncoder, bitstream_path: str, hls_sig_blksize: int):
    start_time = time.time()

    # ======================== GOP HEADER ======================== #
    # Write the header
    header_path = f'{bitstream_path}_gop_header'
    write_gop_header(video_encoder, header_path)

    # Concatenate everything inside a single file
    subprocess.call(f'rm -f {bitstream_path}', shell=True)
    subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
    subprocess.call(f'rm -f {header_path}', shell=True)
    # ======================== GOP HEADER ======================== #

    for idx_coding_order in range(video_encoder.coding_structure.get_number_of_frames()):
        frame = video_encoder.coding_structure.get_frame_from_coding_order(idx_coding_order)
        # assert frame.already_encoded, f'Frame {frame.display_order} has not been encoded yet!'

        # Retrieve the frame encoder corresponding to the frame
        frame_encoder, _ = video_encoder.all_frame_encoders.get(str(idx_coding_order))

        frame_bitstream_path = f'{bitstream_path}_{idx_coding_order}'
        encode_frame(
            video_encoder,
            frame_encoder,
            frame_bitstream_path,
            idx_coding_order,
            hls_sig_blksize
        )
        subprocess.call(f'cat {frame_bitstream_path} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {frame_bitstream_path}', shell=True)

    real_rate_byte = os.path.getsize(bitstream_path)
    # Not very elegant but look at the first frame cool-chic to get the video resolution
    h, w = video_encoder.all_frame_encoders["0"][0].coolchic_encoder_param.img_size
    real_rate_bpp = (
        real_rate_byte * 8 / (h * w * len(video_encoder.coding_structure.frames))
    )
    print(f'Real rate        [kBytes]: {real_rate_byte / 1000:9.3f}')
    print(f'Real rate           [bpp]: {real_rate_bpp :9.3f}')

    elapsed = time.time() - start_time
    print(f'Encoding time: {elapsed:4.3f} sec')


@torch.no_grad()
def encode_frame(
    video_encoder: VideoEncoder,
    frame_encoder: FrameEncoder,
    bitstream_path: str,
    idx_coding_order: int,
    hls_sig_blksize: int
):

    """Convert a model to a bitstream located at <bitstream_path>.

    Args:
        model (CoolChicEncoder): A trained and quantized model
        bitstream_path (str): Where to save the bitstream
    """

    torch.use_deterministic_algorithms(True)
    frame_encoder.set_to_eval()
    frame_encoder.to_device('cpu')

    subprocess.call(f'rm -f {bitstream_path}', shell=True)

    # Load the references
    current_frame = video_encoder.coding_structure.get_frame_from_coding_order(
        idx_coding_order
    )
    current_frame.refs_data = video_encoder.get_ref_data(current_frame)
    current_frame.upsample_reference_to_444()

    # Move to pure-int Arm.  Transfer the quantized weights from the fp Arm.
    arm_fp_param = frame_encoder.coolchic_encoder.arm.get_param()
    print("recovered arm params", arm_fp_param.keys())
    arm_int = ArmInt(
        frame_encoder.coolchic_encoder.param.dim_arm,
        frame_encoder.coolchic_encoder.param.n_hidden_layers_arm,
        FIXED_POINT_FRACTIONAL_MULT,
        pure_int=True
    )
    frame_encoder.coolchic_encoder.arm = arm_int
    frame_encoder.coolchic_encoder.arm.set_param_from_float(arm_fp_param)
    print("set armint(pureint) params")

    # ================= Encode the MLP into a bitstream file ================ #
    ac_max_val_nn = get_ac_max_val_nn(frame_encoder)

    scale_index_nn: DescriptorCoolChic = {}
    q_step_index_nn: DescriptorCoolChic = {}
    n_bytes_nn: DescriptorCoolChic = {}
    for cur_module_name in frame_encoder.coolchic_encoder.modules_to_send:
        # Prepare to store values dedicated to the current modules
        scale_index_nn[cur_module_name] = {}
        q_step_index_nn[cur_module_name] = {}
        n_bytes_nn[cur_module_name] = {}

        module_to_encode = getattr(frame_encoder.coolchic_encoder, cur_module_name)

        weights, bias = [], []
        # Retrieve all the weights and biases for the ARM MLP
        q_step_index_nn[cur_module_name]['weight'] = -1
        q_step_index_nn[cur_module_name]['bias'] = -1
        for k, v in module_to_encode.named_parameters():
            assert cur_module_name in ['arm', 'synthesis', 'upsampling'], f'Unknow module name {cur_module_name}. '\
                'Module name should be in ["arm", "synthesis", "upsampling"].'

            Q_STEPS = POSSIBLE_Q_STEP.get(cur_module_name)

            if k.endswith(".w") or k.endswith(".weight"):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_possible_q_step = POSSIBLE_Q_STEP.get(cur_module_name).get("weight")
                cur_q_step = frame_encoder.coolchic_encoder.nn_q_step.get(
                    cur_module_name
                ).get("weight")
                cur_q_step_index = int(
                    torch.argmin((cur_possible_q_step - cur_q_step).abs()).item()
                )

                # Store it into q_step_index_nn. It is overwritten for each
                # loop but it does not matter
                q_step_index_nn[cur_module_name]["weight"] = cur_q_step_index

                # Quantize the weight with the actual quantization step and add it
                # to the list of (quantized) weights
                # print(cur_module_name, k, v)
                if cur_module_name == "arm":
                    # Our weights are stored as fixed point, we use shifts to get the integer values of quantized results.
                    # Our int vals are int(floatval << FPFBITS)
                    q_step_shift = abs(POSSIBLE_Q_STEP_SHIFT["arm"]["weight"][cur_q_step_index])
                    delta = int(FIXED_POINT_FRACTIONAL_BITS - q_step_shift)
                    if delta > 0:
                        pos_v = (v >> delta)     # a following <<delta would be the actual weight.
                        neg_v = -((-v >> delta)) # a following <<delta would be the actual weight.
                        v = torch.where(v < 0, neg_v, pos_v)
                    weights.append(v.flatten())
                else:
                    weights.append(
                        torch.round(v / cur_possible_q_step[cur_q_step_index]).flatten()
                    )

            elif k.endswith(".b") or k.endswith(".bias"):
                # Find the index of the closest quantization step in the list of
                # the Q_STEPS quantization step.
                cur_possible_q_step = POSSIBLE_Q_STEP.get(cur_module_name).get("bias")
                cur_q_step = frame_encoder.coolchic_encoder.nn_q_step.get(
                    cur_module_name
                ).get("bias")
                cur_q_step_index = int(
                    torch.argmin((cur_possible_q_step - cur_q_step).abs()).item()
                )

                # Store it into q_step_index_nn. It is overwritten for each
                # loop but it does not matter
                q_step_index_nn[cur_module_name]["bias"] = cur_q_step_index

                # Quantize the bias with the actual quantization step and add it
                # to the list of (quantized) bias
                if cur_module_name == "arm":
                    # Our biases are stored as fixed point, we use shifts to get the integer values of quantized results.
                    # Our int vals are int(floatval << FPFBITS << FPFBITS)
                    q_step_shift = abs(POSSIBLE_Q_STEP_SHIFT["arm"]["bias"][cur_q_step_index])
                    delta = int(FIXED_POINT_FRACTIONAL_BITS*2 - q_step_shift)
                    if delta > 0:
                        pos_v = (v >> delta)     # a following <<delta would be the actual weight.
                        neg_v = -((-v >> delta)) # a following <<delta would be the actual weight.
                        v = torch.where(v < 0, neg_v, pos_v)
                    bias.append(v.flatten())
                else:
                    bias.append(
                        torch.round(v / cur_possible_q_step[cur_q_step_index]).flatten()
                    )

        # Gather them
        weights = torch.cat(weights).flatten()
        have_bias = len(bias) != 0
        if have_bias:
            bias = torch.cat(bias).flatten()

        # ----------------- Actual entropy coding
        # It happens on cpu
        weights = weights.cpu()
        if have_bias:
            bias = bias.cpu()

        cur_bitstream_path = f'{bitstream_path}_{cur_module_name}_weight'

        # either code directly (normal), or search for best (backwards compatible).
        scale_index_weight = frame_encoder.coolchic_encoder.nn_expgol_cnt[cur_module_name]['weight']
        if scale_index_weight is None:
            scale_index_weight = -1 # Search for best.
        scale_index_weight = \
            cc_code_wb_bac(cur_bitstream_path,
                        weights.flatten().to(torch.int32).tolist(),
                        scale_index_weight # search for best count if -1
                       )
        scale_index_nn[cur_module_name]['weight'] = scale_index_weight

        n_bytes_nn[cur_module_name]['weight'] = os.path.getsize(cur_bitstream_path)

        if have_bias:
            cur_bitstream_path = f'{bitstream_path}_{cur_module_name}_bias'

            # either code directly (normal), or search for best (backwards compatible).
            scale_index_bias = frame_encoder.coolchic_encoder.nn_expgol_cnt[cur_module_name]['bias']
            if scale_index_bias is None:
                scale_index_bias = -1 # Search for best.
            scale_index_bias = \
            cc_code_wb_bac(cur_bitstream_path,
                        bias.flatten().to(torch.int32).tolist(),
                        scale_index_bias # search for best count if -1
                       )
            scale_index_nn[cur_module_name]['bias'] = scale_index_bias

            n_bytes_nn[cur_module_name]['bias'] = os.path.getsize(cur_bitstream_path)
        else:
            n_bytes_nn[cur_module_name]['bias'] = 0
    # ================= Encode the MLP into a bitstream file ================ #

    # =============== Encode the latent into a bitstream file =============== #
    # To ensure perfect reproducibility between the encoder and the decoder,
    # we load the the different sub-networks from the bitstream here.
    for module_name in frame_encoder.coolchic_encoder.modules_to_send:
        assert module_name in ['arm', 'synthesis', 'upsampling'], f'Unknow module name {module_name}. '\
            'Module name should be in ["arm", "synthesis", "upsampling"].'

        if module_name == 'arm':
            empty_module = ArmInt(
                frame_encoder.coolchic_encoder.param.dim_arm,
                frame_encoder.coolchic_encoder.param.n_hidden_layers_arm,
                FIXED_POINT_FRACTIONAL_MULT,
                pure_int = True
            )
        elif module_name == 'synthesis':
            empty_module =  Synthesis(
                sum(frame_encoder.coolchic_encoder.param.n_ft_per_res),
                frame_encoder.coolchic_encoder.param.layers_synthesis
            )
        elif module_name == 'upsampling':
            empty_module = Upsampling(
                    frame_encoder.coolchic_encoder.param.upsampling_kernel_size,
                    frame_encoder.coolchic_encoder.param.static_upsampling_kernel
                )

        Q_STEPS = POSSIBLE_Q_STEP.get(module_name)

        have_bias = q_step_index_nn[module_name].get('bias') >= 0
        loaded_module = decode_network(
            empty_module,
            DescriptorNN(
                weight = f'{bitstream_path}_{module_name}_weight',
                bias = f'{bitstream_path}_{module_name}_bias' if have_bias else "",
            ),
            DescriptorNN (
                weight=Q_STEPS["weight"][q_step_index_nn[module_name]["weight"]],
                bias=Q_STEPS["bias"][q_step_index_nn[module_name]["bias"]]
            ),
            DescriptorNN (
                scale_index_nn[module_name]["weight"],
                bias=(
                    scale_index_nn[module_name]["bias"]
                )
                if have_bias
                else 0,
            ),
            ac_max_val_nn
        )
        setattr(frame_encoder.coolchic_encoder, module_name, loaded_module)

    frame_encoder.coolchic_encoder.to_device('cpu')
    frame_encoder.set_to_eval()

    ac_max_val_latent = get_ac_max_val_latent(frame_encoder)

    # Setting visu to true allows to recover 2D mu, scale and latents

    encoder_output = frame_encoder.forward(
        reference_frames=[ref_i.data for ref_i in current_frame.refs_data],
        quantizer_noise_type="noise",
        quantizer_type="hardround",
        AC_MAX_VAL=ac_max_val_latent,
        flag_additional_outputs=True,
    )

    # Encode the different 2d latent grids one after the other
    n_bytes_per_latent = []
    torch.set_printoptions(threshold=10000000)
    ctr_2d_ft = 0
    # Loop on the different resolutions
    for index_lat_resolution in range(frame_encoder.coolchic_encoder.param.latent_n_grids):
        current_mu = encoder_output.additional_data.get('detailed_mu')[index_lat_resolution]
        current_scale = encoder_output.additional_data.get('detailed_scale')[index_lat_resolution]
        current_log_scale = encoder_output.additional_data.get('detailed_log_scale')[index_lat_resolution]
        current_y = encoder_output.additional_data.get('detailed_sent_latent')[index_lat_resolution]

        c_i, h_i, w_i = current_y.size()[-3:]

        # Nothing to send!
        if c_i == 0:
            n_bytes_per_latent.append(0)
            cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
            # Still create an empty file for coherence
            subprocess.call(f'touch {cur_latent_bitstream}', shell=True)
            ctr_2d_ft += 1
            continue

        # Loop on the different 2D grids composing one resolutions
        for index_lat_feature in range(c_i):
            y_this_ft = current_y[:, index_lat_feature, :, :].flatten().cpu()
            mu_this_ft = current_mu[:, index_lat_feature, :, :].flatten().cpu()
            scale_this_ft = current_scale[:, index_lat_feature, :, :].flatten().cpu()
            log_scale_this_ft = current_log_scale[:, index_lat_feature, :, :].flatten().cpu()

            if y_this_ft.abs().max() == 0:
                n_bytes_per_latent.append(0)
                cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
                # Still create an empty file for coherence
                subprocess.call(f'touch {cur_latent_bitstream}', shell=True)
                ctr_2d_ft += 1
                continue

            cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
            cc_code_latent_layer_bac(
                cur_latent_bitstream,
                y_this_ft.flatten().to(torch.int32).tolist(),
                (mu_this_ft*FIXED_POINT_FRACTIONAL_MULT).round().flatten().to(torch.int32).tolist(),
                (log_scale_this_ft*FIXED_POINT_FRACTIONAL_MULT).round().flatten().to(torch.int32).tolist(),
                h_i, w_i,
                hls_sig_blksize,
            )
            n_bytes_per_latent.append(os.path.getsize(cur_latent_bitstream))

            ctr_2d_ft += 1

    # Write the header
    header_path = f'{bitstream_path}_header'
    write_frame_header(
        frame_encoder,
        current_frame,
        header_path,
        n_bytes_per_latent,
        q_step_index_nn,
        scale_index_nn,
        n_bytes_nn,
        ac_max_val_nn,
        ac_max_val_latent,
        hls_sig_blksize,
    )

    # Concatenate everything inside a single file
    subprocess.call(f'rm -f {bitstream_path}', shell=True)
    subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
    subprocess.call(f'rm -f {header_path}', shell=True)

    for cur_module_name in ['arm', 'upsampling', 'synthesis']:
        for parameter_type in ['weight', 'bias']:
            cur_bitstream = f'{bitstream_path}_{cur_module_name}_{parameter_type}'
            if os.path.exists(cur_bitstream):
                subprocess.call(f'cat {cur_bitstream} >> {bitstream_path}', shell=True)
                subprocess.call(f'rm -f {cur_bitstream}', shell=True)

    ctr_2d_ft = 0
    for index_lat_resolution in range(frame_encoder.coolchic_encoder.param.latent_n_grids):

        # No feature: still increment the counter and remove the temporary bitstream file
        if frame_encoder.coolchic_encoder.latent_grids[index_lat_resolution].size()[1] == 0:
            cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
            subprocess.call(f'rm -f {cur_latent_bitstream}', shell=True)
            ctr_2d_ft += 1

        for index_lat_feature in range(frame_encoder.coolchic_encoder.latent_grids[index_lat_resolution].size()[1]):
            cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
            subprocess.call(f'cat {cur_latent_bitstream} >> {bitstream_path}', shell=True)
            subprocess.call(f'rm -f {cur_latent_bitstream}', shell=True)
            ctr_2d_ft += 1


    # Encoding's done, we no longer need deterministic algorithms
    torch.use_deterministic_algorithms(False)
