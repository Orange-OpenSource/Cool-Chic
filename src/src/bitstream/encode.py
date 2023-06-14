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

from bitstream.decode import decode_network
from bitstream.header import write_header
from bitstream.range_coder import RangeCoder
from models.arm import Arm
from models.upsampling import Upsampling
from models.cool_chic import CoolChicEncoder, to_device
from models.synthesis import Synthesis
from utils.constants import POSSIBLE_Q_STEP_ARM_NN, POSSIBLE_Q_STEP_SYN_NN, POSSIBLE_SCALE_NN, FIXED_POINT_FRACTIONAL_MULT
from utils.data_structure import DescriptorNN, DescriptorCoolChic


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

    for cur_module_name in ['arm', 'upsampling', 'synthesis']:
        module_to_encode = getattr(model, cur_module_name)

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
                    (Q_STEPS - q_step_nn[f'{cur_module_name}_weight']).abs()
                ).item())

                # Quantize the weight with the actual quantization step and add it
                # to the list of (quantized) weights
                model_param_quant.append(torch.round(v / Q_STEPS[cur_q_step_index]).flatten())

            elif k.endswith('.b') or k.endswith('.bias'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step_index = int(torch.argmin(
                    (Q_STEPS - q_step_nn[f'{cur_module_name}_bias']).abs()
                ).item())

                # Quantize the bias with the actual quantization step and add it
                # to the list of (quantized) bias
                model_param_quant.append(torch.round(v / Q_STEPS[cur_q_step_index]).flatten())

    # Gather them
    model_param_quant = torch.cat(model_param_quant).flatten()

    # Compute AC_MAX_VAL.
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
    encoder_output = model.forward(visu = True, AC_MAX_VAL = -1)
    latent = torch.cat([y_i.flatten() for y_i in encoder_output.get('2d_y_latent')], dim=0)

    # Compute AC_MAC_VAL
    AC_MAX_VAL = int(torch.ceil(latent.abs().max() + 2).item())
    return AC_MAX_VAL


@torch.no_grad()
def encode(model: CoolChicEncoder, bitstream_path: str, q_step_nn: DescriptorCoolChic):
    """Convert a model to a bistream located at <bistream_path>. The model is a quantized
    model but in floating point e.g. 1.25 if q_step is 0.25. Consequently, the quantization
    is provided to transform the NN weights into integer during their entropy coding.

    Args:
        model (CoolChicEncoder): A trained and quantized model
        bitstream_path (str): Where to save the bitstream
        q_step_nn (DescriptorCoolChic): Describe the quantization steps used
            for the weight and bias of the network.
    """

    start_time = time.time()
    # Ensure encoding/decoding replicability on different hardware
    torch.use_deterministic_algorithms(True)
    # Run on CPU for more reliability (particularly the ARM module)
    model = model.eval()
    model = to_device(model, 'cpu')

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
    for cur_module_name in ['arm', 'upsampling', 'synthesis']:
        # Prepare to store values dedicated to the current modules
        scale_index_nn[cur_module_name] = {}
        q_step_index_nn[cur_module_name] = {}
        n_bytes_nn[cur_module_name] = {}

        module_to_encode = getattr(model, cur_module_name)

        weights, bias = [], []
        # Retrieve all the weights and biases for the ARM MLP
        q_step_index_nn[cur_module_name]['weight'] = -1
        q_step_index_nn[cur_module_name]['bias'] = -1
        for k, v in module_to_encode.named_parameters():
            if cur_module_name == 'arm':
                Q_STEPS = POSSIBLE_Q_STEP_ARM_NN
            else:
                Q_STEPS = POSSIBLE_Q_STEP_SYN_NN

            if k.endswith('.w') or k.endswith('.weight'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step_index = int(torch.argmin(
                    (Q_STEPS - q_step_nn[f'{cur_module_name}_weight']).abs()
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
                    (Q_STEPS - q_step_nn[f'{cur_module_name}_bias']).abs()
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
    for module_name in ['arm', 'upsampling', 'synthesis']:
        if module_name == 'arm':
            empty_module = Arm(model.non_zero_pixel_ctx, model.param.layers_arm)
            Q_STEPS = POSSIBLE_Q_STEP_ARM_NN
        elif module_name == 'synthesis':
            empty_module =  Synthesis(model.param.latent_n_grids, model.param.layers_synthesis)
            Q_STEPS = POSSIBLE_Q_STEP_SYN_NN
        else:
            empty_module = Upsampling(model.param.upsampling_kernel_size)
            Q_STEPS = POSSIBLE_Q_STEP_SYN_NN

        have_bias = q_step_index_nn[module_name]['bias'] >= 0
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
        setattr(model, module_name, loaded_module)

    model.arm.set_quant(FIXED_POINT_FRACTIONAL_MULT)
    # model.non_zero_pixel_ctx_index = model.non_zero_pixel_ctx_index.to('cpu')
    model = to_device(model, 'cpu')

    ac_max_val_latent = get_ac_max_val_latent(model)
    range_coder_latent = RangeCoder(model.param.n_ctx_rowcol, ac_max_val_latent)

    # Setting visu to true allows to recover 2D mu, scale and latents
    encoder_output = model.forward(visu = True, AC_MAX_VAL = range_coder_latent.AC_MAX_VAL)

    # Encode the different latent grids one after the other
    n_bytes_per_latent = []
    torch.set_printoptions(threshold=10000000)
    for i in range(model.param.latent_n_grids):
        current_mu = encoder_output.get('2d_y_mu')[i]
        current_scale = encoder_output.get('2d_y_scale')[i]
        current_y = encoder_output.get('2d_y_latent')[i]

        # Nothing to send!
        if current_y.abs().max() == 0:
            n_bytes_per_latent.append(0)
            continue

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

    for i in range(model.param.latent_n_grids):
        if n_bytes_per_latent[i] > 0:
            cur_latent_bitstream = f'{bitstream_path}_{i}'
            subprocess.call(f'cat {cur_latent_bitstream} >> {bitstream_path}', shell=True)
            subprocess.call(f'rm -f {cur_latent_bitstream}', shell=True)

    real_rate_byte = os.path.getsize(bitstream_path)
    real_rate_bpp = real_rate_byte * 8 / (model.param.img_size[-1] * model.param.img_size[-2])
    print(f'Real rate        [kBytes]: {real_rate_byte / 1000:9.3f}')
    print(f'Real rate           [bpp]: {real_rate_bpp :9.3f}')

    elapsed = time.time() - start_time
    print(f'Encoding time: {elapsed:4.3f} sec')

    # Encoding's done, we no longer need deterministic algorithms
    torch.use_deterministic_algorithms(False)
