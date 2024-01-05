# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import math
import os
import subprocess
import torch
import time

from bitstream.decode import decode_network, get_sub_bitstream_path
from bitstream.header import write_gop_header, write_frame_header
from bitstream.range_coder import RangeCoder
from models.coolchic_components.arm import Arm
from models.coolchic_components.upsampling import Upsampling
from models.coolchic_components.synthesis import Synthesis
from models.frame_encoder import FrameEncoder
from models.video_encoder import VideoEncoder
from utils.misc import POSSIBLE_Q_STEP_ARM_NN, POSSIBLE_Q_STEP_SYN_NN, POSSIBLE_Q_STEP_UPS_NN, POSSIBLE_SCALE_NN, FIXED_POINT_FRACTIONAL_MULT, DescriptorNN, DescriptorCoolChic


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
            if cur_module_name == 'arm':
                Q_STEPS = POSSIBLE_Q_STEP_ARM_NN
            else:
                Q_STEPS = POSSIBLE_Q_STEP_SYN_NN

            if k.endswith('.w') or k.endswith('.weight'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step_index = int(torch.argmin(
                    (Q_STEPS - module_to_encode._q_step.get('weight')).abs()
                ).item())

                # Quantize the weight with the actual quantization step and add it
                # to the list of (quantized) weights
                model_param_quant.append(torch.round(v / Q_STEPS[cur_q_step_index]).flatten())

            elif k.endswith('.b') or k.endswith('.bias'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step_index = int(torch.argmin(
                    (Q_STEPS - module_to_encode._q_step.get('bias')).abs()
                ).item())

                # Quantize the bias with the actual quantization step and add it
                # to the list of (quantized) bias
                model_param_quant.append(torch.round(v / Q_STEPS[cur_q_step_index]).flatten())

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
        flag_additional_outputs=True, AC_MAX_VAL=-1, use_ste_quant=True
    )
    latent = torch.cat(
        [
            y_i.flatten()
            for y_i in encoder_output.get('additional_data').get('detailed_sent_latent')
        ],
        dim=0
    )

    # Compute AC_MAC_VAL
    AC_MAX_VAL = int(torch.ceil(latent.abs().max() + 2).item())
    return AC_MAX_VAL


def encode_video(video_encoder: VideoEncoder, bitstream_path: str):
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
        frame_encoder = video_encoder.all_frame_encoders[
            video_encoder.get_key_all_frame_encoders(idx_coding_order, 'best')
        ]

        frame_bitstream_path = f'{bitstream_path}_{idx_coding_order}'
        encode_frame(video_encoder, frame_encoder, frame_bitstream_path)
        subprocess.call(f'cat {frame_bitstream_path} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {frame_bitstream_path}', shell=True)

    real_rate_byte = os.path.getsize(bitstream_path)
    h, w = video_encoder.coding_structure.frames[0].data.img_size
    real_rate_bpp = real_rate_byte * 8 / (h * w * len(video_encoder.coding_structure.frames))
    print(f'Real rate        [kBytes]: {real_rate_byte / 1000:9.3f}')
    print(f'Real rate           [bpp]: {real_rate_bpp :9.3f}')

    elapsed = time.time() - start_time
    print(f'Encoding time: {elapsed:4.3f} sec')


@torch.no_grad()
def encode_frame(video_encoder: VideoEncoder, frame_encoder: FrameEncoder, bitstream_path: str):
    """Convert a model to a bitstream located at <bitstream_path>.

    Args:
        model (CoolChicEncoder): A trained and quantized model
        bitstream_path (str): Where to save the bitstream
    """

    torch.use_deterministic_algorithms(True)
    frame_encoder.set_to_eval()
    frame_encoder.to_device('cpu')

    subprocess.call(f'rm -f {bitstream_path}', shell=True)

    # Load the frame and its reference
    frame_encoder.frame = video_encoder.load_data_and_refs(frame_encoder.frame)

    # ================= Encode the MLP into a bitstream file ================ #
    ac_max_val_nn = get_ac_max_val_nn(frame_encoder)
    range_coder_nn = RangeCoder(
        0,     # 0 because we don't have a ctx_row_col
        ac_max_val_nn
    )

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

            if cur_module_name == 'arm':
                Q_STEPS = POSSIBLE_Q_STEP_ARM_NN
            elif cur_module_name == 'synthesis':
                Q_STEPS = POSSIBLE_Q_STEP_SYN_NN
            elif cur_module_name == 'upsampling':
                Q_STEPS = POSSIBLE_Q_STEP_UPS_NN

            if k.endswith('.w') or k.endswith('.weight'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step_index = int(torch.argmin(
                    (Q_STEPS - module_to_encode._q_step.get('weight')).abs()
                ).item())

                # Store it into q_step_index_nn. It is overwritten for each
                # loop but it does not matter
                q_step_index_nn[cur_module_name]['weight'] = cur_q_step_index

                # Quantize the weight with the actual quantization step and add it
                # to the list of (quantized) weights
                weights.append(torch.round(v / Q_STEPS[cur_q_step_index]).flatten())

            elif k.endswith('.b') or k.endswith('.bias'):
                # Find the index of the closest quantization step in the list of
                # the Q_STEPS quantization step.
                cur_q_step_index = int(torch.argmin(
                    (Q_STEPS - module_to_encode._q_step.get('bias')).abs()
                ).item())

                # Store it into q_step_index_nn. It is overwritten for each
                # loop but it does not matter
                q_step_index_nn[cur_module_name]['bias'] = cur_q_step_index

                # Quantize the bias with the actual quantization step and add it
                # to the list of (quantized) bias
                bias.append(torch.round(v / Q_STEPS[cur_q_step_index]).flatten())

        # Gather them
        weights = torch.cat(weights).flatten()
        have_bias = len(bias) != 0
        if have_bias:
            bias = torch.cat(bias).flatten()

        floating_point_scale_weight = weights.std().item() / math.sqrt(2)
        if have_bias:
            floating_point_scale_bias = bias.std().item() / math.sqrt(2)

        # Find the closest element to the actual scale in the POSSIBLE_SCALE_NN list
        scale_index_weight = int(
            torch.argmin((POSSIBLE_SCALE_NN - floating_point_scale_weight).abs()).item()
        )
        if have_bias:
            scale_index_bias = int(
                torch.argmin((POSSIBLE_SCALE_NN - floating_point_scale_bias).abs()).item()
            )
        # Store this information for the header
        scale_index_nn[cur_module_name]['weight'] = scale_index_weight
        scale_index_nn[cur_module_name]['bias'] = scale_index_bias if have_bias else -1

        scale_weight = POSSIBLE_SCALE_NN[scale_index_weight]
        if scale_index_bias >= 0:
            scale_bias = POSSIBLE_SCALE_NN[scale_index_bias]

        # ----------------- Actual entropy coding
        # It happens on cpu
        weights = weights.cpu()
        if have_bias:
            bias = bias.cpu()

        cur_bitstream_path = f'{bitstream_path}_{cur_module_name}_weight'
        range_coder_nn.encode(
            cur_bitstream_path,
            weights,
            torch.zeros_like(weights),
            scale_weight * torch.ones_like(weights),
            CHW = None,     # No wavefront coding for the weights
        )

        n_bytes_nn[cur_module_name]['weight'] = os.path.getsize(cur_bitstream_path)

        if have_bias:
            cur_bitstream_path = f'{bitstream_path}_{cur_module_name}_bias'
            range_coder_nn.encode(
                cur_bitstream_path,
                bias,
                torch.zeros_like(bias),
                scale_bias * torch.ones_like(bias),
                CHW = None,     # No wavefront coding for the bias
            )
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
            empty_module = Arm(
                frame_encoder.coolchic_encoder.non_zero_pixel_ctx,
                frame_encoder.coolchic_encoder.param.layers_arm
            )
            Q_STEPS = POSSIBLE_Q_STEP_ARM_NN
        elif module_name == 'synthesis':
            empty_module =  Synthesis(
                sum(frame_encoder.coolchic_encoder.param.n_ft_per_res),
                frame_encoder.coolchic_encoder.param.layers_synthesis
            )
            Q_STEPS = POSSIBLE_Q_STEP_SYN_NN
        elif module_name == 'upsampling':
            empty_module = Upsampling(frame_encoder.coolchic_encoder.param.upsampling_kernel_size)
            Q_STEPS = POSSIBLE_Q_STEP_SYN_NN

        have_bias = q_step_index_nn[module_name].get('bias') >= 0
        loaded_module = decode_network(
            empty_module,
            DescriptorNN(
                weight = f'{bitstream_path}_{module_name}_weight',
                bias = f'{bitstream_path}_{module_name}_bias' if have_bias else "",
            ),
            DescriptorNN (
                weight = Q_STEPS[q_step_index_nn[module_name]['weight']],
                bias = Q_STEPS[q_step_index_nn[module_name]['bias']] if have_bias else 0,
            ),
            DescriptorNN (
                weight = POSSIBLE_SCALE_NN[scale_index_nn[module_name]['weight']],
                bias = POSSIBLE_SCALE_NN[scale_index_nn[module_name]['bias']] if have_bias else 0,
            ),
            ac_max_val_nn
        )
        setattr(frame_encoder.coolchic_encoder, module_name, loaded_module)

    frame_encoder.coolchic_encoder.arm.set_quant(FIXED_POINT_FRACTIONAL_MULT)
    frame_encoder.coolchic_encoder.to_device('cpu')

    ac_max_val_latent = get_ac_max_val_latent(frame_encoder)
    range_coder_latent = RangeCoder(frame_encoder.coolchic_encoder.param.n_ctx_rowcol, ac_max_val_latent)

    # Setting visu to true allows to recover 2D mu, scale and latents
    encoder_output = frame_encoder.forward(
        flag_additional_outputs=True, AC_MAX_VAL=range_coder_latent.AC_MAX_VAL, use_ste_quant=True
    )

    # Encode the different 2d latent grids one after the other
    n_bytes_per_latent = []
    torch.set_printoptions(threshold=10000000)
    # Loop on the different resolutions
    for index_lat_resolution in range(frame_encoder.coolchic_encoder.param.latent_n_grids):
        current_mu = encoder_output.additional_data.get('detailed_mu')[index_lat_resolution]
        current_scale = encoder_output.additional_data.get('detailed_scale')[index_lat_resolution]
        current_y = encoder_output.additional_data.get('detailed_sent_latent')[index_lat_resolution]

        c_i, h_i, w_i = current_y.size()[-3:]
        
        # Nothing to send!
        if c_i == 0:
            n_bytes_per_latent.append(0)
            continue

        # Loop on the different 2D grids composing one resolutions
        for index_lat_feature in range(c_i):
            y_this_ft = current_y[:, index_lat_feature, :, :].flatten().cpu()
            mu_this_ft = current_mu[:, index_lat_feature, :, :].flatten().cpu()
            scale_this_ft = current_scale[:, index_lat_feature, :, :].flatten().cpu()

            if y_this_ft.abs().max() == 0:
                n_bytes_per_latent.append(0)
                continue

            cur_latent_bitstream = get_sub_bitstream_path(
                bitstream_path, index_lat_resolution, index_lat_feature
            )
            range_coder_latent.encode(
                cur_latent_bitstream, y_this_ft, mu_this_ft, scale_this_ft, (1, h_i, w_i)
            )
            n_bytes_per_latent.append(os.path.getsize(cur_latent_bitstream))

    # Write the header
    header_path = f'{bitstream_path}_header'
    write_frame_header(
        frame_encoder,
        header_path,
        n_bytes_per_latent,
        q_step_index_nn,
        scale_index_nn,
        n_bytes_nn,
        ac_max_val_nn,
        ac_max_val_latent,
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
        for index_lat_feature in range(frame_encoder.coolchic_encoder.latent_grids[index_lat_resolution].size()[1]):
            cur_latent_bitstream = get_sub_bitstream_path(
                bitstream_path, index_lat_resolution, index_lat_feature
            )
            subprocess.call(f'cat {cur_latent_bitstream} >> {bitstream_path}', shell=True)
            subprocess.call(f'rm -f {cur_latent_bitstream}', shell=True)
            ctr_2d_ft += 1


    # Encoding's done, we no longer need deterministic algorithms
    torch.use_deterministic_algorithms(False)
