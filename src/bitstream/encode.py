# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
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
from typing import Dict
from torch import Tensor
from bitstream.decode import decode_network

from bitstream.header import DescriptorCoolChic, DescriptorNN, write_header
from bitstream.range_coder import RangeCoder
from models.arm import ArmMLP
from models.cool_chic import CoolChicEncoder
from models.synthesis import SynthesisMLP
from utils.constants import POSSIBLE_Q_STEP_NN, POSSIBLE_SCALE_NN


def get_ac_max_val_nn(model: CoolChicEncoder, q_step_nn: DescriptorCoolChic) -> int:
    """Return the maximum amplitude of the quantized model (i.e. weight / q_step).
    This allows to get the AC_MAX_VAL constant, used by the entropy coder. All
    symbols sent by the entropy coder will be in [-AC_MAX_VAL, AC_MAX_VAL - 1].

    Args:
        mode (CoolChicEncoder): Model to quantize.
        q_step_nn (DescriptorCoolChic): Quantization step of the different NNs.
            See in header.py for the details.

    Returns:
        int: The AC_MAX_VAL parameter.
    """
    model_param_quant = []

    for cur_module_name in ['arm', 'synthesis']:
        module_to_encode = getattr(model, cur_module_name)

        # Retrieve all the weights and biases for the ARM MLP
        for k, v in module_to_encode.named_parameters():
            if 'mlp' not in k:
                continue

            if k.endswith('.w'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step_index = int(torch.argmin(
                    (POSSIBLE_Q_STEP_NN - q_step_nn[f'{cur_module_name}_weight']).abs()
                ).item())

                # Quantize the weight with the actual quantization step and add it
                # to the list of (quantized) weights
                model_param_quant.append(torch.round(v / POSSIBLE_Q_STEP_NN[cur_q_step_index]).flatten())

            elif k.endswith('.b'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step_index = int(torch.argmin(
                    (POSSIBLE_Q_STEP_NN - q_step_nn[f'{cur_module_name}_bias']).abs()
                ).item())

                # Quantize the bias with the actual quantization step and add it
                # to the list of (quantized) bias
                model_param_quant.append(torch.round(v / POSSIBLE_Q_STEP_NN[cur_q_step_index]).flatten())

    # Gather them
    model_param_quant = torch.cat(model_param_quant).flatten()

    # Compute AC_MAC_VAL
    AC_MAX_VAL = int(torch.ceil(model_param_quant.abs().max() + 2).item())
    return AC_MAX_VAL


def get_ac_max_val_latent(model: CoolChicEncoder) -> int:
    """Return the maximum amplitude of the quantized latent variables.
    This allows to get the AC_MAX_VAL constant, used by the entropy coder. All
    symbols sent by the entropy coder will be in [-AC_MAX_VAL, AC_MAX_VAL - 1].

    Args:
        mode (CoolChicEncoder): Model storing the latent.

    Returns:
        int: The AC_MAX_VAL parameter.
    """
    # Setting visu to true allows to recover 2D mu, scale and latents
    # Don't specify AC_MAX_VAL now: we let the latents evolve freely to capture
    # their dynamic.
    encoder_output = model.forward(get_proba_param = True, AC_MAX_VAL = -1)
    latent = torch.cat([y_i.flatten() for y_i in encoder_output.get('latent')], dim=0)

    # Compute AC_MAC_VAL
    AC_MAX_VAL = int(torch.ceil(latent.abs().max() + 2).item())
    return AC_MAX_VAL


@torch.no_grad()
def encode(
    model: CoolChicEncoder,
    bitstream_path: str,
    q_step_nn: DescriptorCoolChic,
) -> int:
    """Encode an image learned by model into a bitstream.

    Args:
        model (CoolChicEncoder): The trained overfitted encoder for the image.
        bitstream_path (str): Where to write the bitstream.
        q_step_nn (DescriptorCoolChic): Quantization step of the different NNs.
            See in header.py for the details.

    Returns:
        int: Rate in **byte**
    """

    start_time = time.time()
    # Ensure encoding/decoding replicability on different hardware
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Run on CPU for more reliability (particularly the ARM module)
    model = model.eval().to('cpu')

    if os.path.exists(bitstream_path):
        print(f'Found an already existing bitstream at {bitstream_path}... deleting!')
        subprocess.call(f'rm {bitstream_path}', shell=True)

    # ================= Encode the MLP into a bitstream file ================ #
    ac_max_val_nn = get_ac_max_val_nn(model, q_step_nn)
    range_coder_nn = RangeCoder(
        0,     # 0 because we don't have a ctx_row_col
        ac_max_val_nn
    )

    scale_index_nn: DescriptorCoolChic = {}
    q_step_index_nn: DescriptorCoolChic = {}
    n_bytes_nn: DescriptorCoolChic = {}
    for cur_module_name in ['arm', 'synthesis']:
        # Prepare to store values dedicated to the current modules
        scale_index_nn[cur_module_name] = {}
        q_step_index_nn[cur_module_name] = {}
        n_bytes_nn[cur_module_name] = {}

        module_to_encode = getattr(model, cur_module_name)

        weights, bias = [], []
        # Retrieve all the weights and biases for the ARM MLP
        for k, v in module_to_encode.named_parameters():
            if not 'mlp' in k:
                continue

            if k.endswith('.w'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step_index = int(torch.argmin(
                    (POSSIBLE_Q_STEP_NN - q_step_nn[f'{cur_module_name}_weight']).abs()
                ).item())

                # Store it into q_step_index_nn. It is overwritten for each
                # loop but it does not matter
                q_step_index_nn[cur_module_name]['weight'] = cur_q_step_index

                # Quantize the weight with the actual quantization step and add it
                # to the list of (quantized) weights
                weights.append(torch.round(v / POSSIBLE_Q_STEP_NN[cur_q_step_index]).flatten())

            elif k.endswith('.b'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step_index = int(torch.argmin(
                    (POSSIBLE_Q_STEP_NN - q_step_nn[f'{cur_module_name}_bias']).abs()
                ).item())

                # Store it into q_step_index_nn. It is overwritten for each
                # loop but it does not matter
                q_step_index_nn[cur_module_name]['bias'] = cur_q_step_index

                # Quantize the bias with the actual quantization step and add it
                # to the list of (quantized) bias
                bias.append(torch.round(v / POSSIBLE_Q_STEP_NN[cur_q_step_index]).flatten())

        # Gather them
        weights = torch.cat(weights).flatten()
        bias = torch.cat(bias).flatten()

        floating_point_scale_weight = weights.std().item() / math.sqrt(2)
        floating_point_scale_bias = bias.std().item() / math.sqrt(2)

        # Find the closest element to the actual scale in the POSSIBLE_SCALE_NN list
        scale_index_weight = int(
            torch.argmin((POSSIBLE_SCALE_NN - floating_point_scale_weight).abs()).item()
        )
        scale_index_bias = int(
            torch.argmin((POSSIBLE_SCALE_NN - floating_point_scale_bias).abs()).item()
        )
        # Store this information for the header
        scale_index_nn[cur_module_name]['weight'] = scale_index_weight
        scale_index_nn[cur_module_name]['bias'] = scale_index_bias

        scale_weight = POSSIBLE_SCALE_NN[scale_index_weight]
        scale_bias = POSSIBLE_SCALE_NN[scale_index_bias]

        # ----------------- Actual entropy coding
        # It happens on cpu
        weights = weights.cpu()
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

        cur_bitstream_path = f'{bitstream_path}_{cur_module_name}_bias'
        range_coder_nn.encode(
            cur_bitstream_path,
            bias,
            torch.zeros_like(bias),
            scale_bias * torch.ones_like(bias),
            CHW = None,     # No wavefront coding for the bias
        )
        n_bytes_nn[cur_module_name]['bias'] = os.path.getsize(cur_bitstream_path)
    # ================= Encode the MLP into a bitstream file ================ #

    # =============== Encode the latent into a bitstream file =============== #
    # To ensure perfect reproducibility between the encoder and the decoder,
    # we load the the different sub-networks from the bitstream here.
    for module_name in ['arm', 'synthesis']:
        if module_name == 'arm':
            empty_module = ArmMLP(model.non_zero_pixel_ctx, model.layers_arm)
        elif module_name == 'synthesis':
            empty_module =  SynthesisMLP(model.latent_n_grids, model.layers_synthesis)

        loaded_module = decode_network(
            empty_module,
            DescriptorNN(
                weight = f'{bitstream_path}_{module_name}_weight',
                bias = f'{bitstream_path}_{module_name}_bias',
            ),
            DescriptorNN (
                weight = POSSIBLE_Q_STEP_NN[q_step_index_nn[module_name]['weight']],
                bias = POSSIBLE_Q_STEP_NN[q_step_index_nn[module_name]['bias']],
            ),
            DescriptorNN (
                weight = POSSIBLE_SCALE_NN[scale_index_nn[module_name]['weight']],
                bias = POSSIBLE_SCALE_NN[scale_index_nn[module_name]['bias']],
            ),
            ac_max_val_nn,
        )
        setattr(model, module_name, loaded_module)

    model.non_zero_pixel_ctx_index = model.non_zero_pixel_ctx_index.to('cpu')

    ac_max_val_latent = get_ac_max_val_latent(model)
    range_coder_latent = RangeCoder(model.n_ctx_rowcol, ac_max_val_latent)

    # Setting visu to true allows to recover 2D mu, scale and latents
    # Use AC_MAX_VAL to clamp the latent so they fit inside the expected range.
    # ! It should not be needed the AC_MAX_VAL parameter is computed to be the maximum
    # ! dynamic of the latent
    encoder_output = model.forward(get_proba_param = True, AC_MAX_VAL = range_coder_latent.AC_MAX_VAL)

    # Encode the different latent grids one after the other
    n_bytes_per_latent = []
    for i in range(model.latent_n_grids):
        current_mu = encoder_output.get('mu')[i]
        current_scale = encoder_output.get('scale')[i]
        current_y = encoder_output.get('latent')[i]

        cur_latent_bitstream = f'{bitstream_path}_{i}'
        range_coder_latent.encode(
            cur_latent_bitstream,
            current_y.flatten().cpu(),
            current_mu.flatten().cpu(),
            current_scale.flatten().cpu(),
            (1, current_y.size()[-2], current_y.size()[-1]),
        )
        n_bytes_per_latent.append(os.path.getsize(cur_latent_bitstream))

    # Write the header
    header_path = f'{bitstream_path}_header'
    write_header(
        model,
        header_path,
        n_bytes_per_latent,
        q_step_index_nn,
        scale_index_nn,
        n_bytes_nn,
        ac_max_val_nn,
        ac_max_val_latent
    )

    # Concatenate everything inside a single file
    subprocess.call(f'rm -f {bitstream_path}', shell=True)
    subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
    subprocess.call(f'rm -f {header_path}', shell=True)

    for cur_module_name in ['arm', 'synthesis']:
        for parameter_type in ['weight', 'bias']:
            cur_bitstream = f'{bitstream_path}_{cur_module_name}_{parameter_type}'
            subprocess.call(f'cat {cur_bitstream} >> {bitstream_path}', shell=True)
            subprocess.call(f'rm -f {cur_bitstream}', shell=True)

    for i in range(model.latent_n_grids):
        cur_latent_bitstream = f'{bitstream_path}_{i}'
        subprocess.call(f'cat {cur_latent_bitstream} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {cur_latent_bitstream}', shell=True)

    real_rate_byte = os.path.getsize(bitstream_path)
    real_rate_bpp = real_rate_byte * 8 / (model.img_size[-1] * model.img_size[-2])
    print(f'Real rate        [kBytes]: {real_rate_byte / 1000:9.3f}')
    print(f'Real rate           [bpp]: {real_rate_bpp :9.3f}')

    elapsed = time.time() - start_time
    print(f'Encoding time: {elapsed:4.3f} sec')
    return real_rate_byte
