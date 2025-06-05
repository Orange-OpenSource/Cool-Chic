# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import os
import subprocess

from enc.component.types import DescriptorCoolChic, DescriptorNN
from enc.nnquant.quantstep import POSSIBLE_Q_STEP, POSSIBLE_Q_STEP_SHIFT
import torch

from dec.nn import decode_network

from enc.bitstream.header import write_frame_header, write_gop_header, cc_latents_zero
from CCLIB.ccencapi import cc_code_latent_layer_bac, cc_code_wb_bac
from enc.bitstream.armint import ArmInt
from enc.component.core.synthesis import Synthesis
from enc.component.core.upsampling import Upsampling
from enc.component.frame import FrameEncoder

# Some constants here
FIXED_POINT_FRACTIONAL_BITS = 8  # 8 works fine in pure int mode
# reduce to 6 for int-in-fp mode
# that has less headroom (23-bit mantissa, not 32)
FIXED_POINT_FRACTIONAL_MULT = 2**FIXED_POINT_FRACTIONAL_BITS


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

            if "weight" in k:
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
                    # No longer relevant without the bi-branch synthesis!
                    # # # Blending -- we get the transformed weight, not the underlying sigmoid parameter.
                    # # # plus: only if >1 branch.
                    # # if cur_module_name == "synthesis" and k.endswith(".parametrizations.weight.original"):
                    # #     if "branch_blender" in k and frame_encoder.coolchic_encoder_param.n_synth_branch == 1:
                    # #         continue # Do not emit unused blender weight.
                    # #     xformed_weights = getattr(module_to_encode, k.replace(".parametrizations.weight.original", "")).weight
                    # #     v = xformed_weights
                    model_param_quant.append(
                        torch.round(v / cur_possible_q_step[cur_q_step_index]).flatten()
                    )


            elif "bias" in k:
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


#def encode_video(video_encoder: VideoEncoder, bitstream_path: str, hls_sig_blksize: int):
#    start_time = time.time()
#
#    # ======================== GOP HEADER ======================== #
#    # Write the header
#    # header_path = f'{bitstream_path}_gop_header' # !!!
#    # write_gop_header(video_encoder, header_path) # !!! NO GOP HEADER HERE WE ARE JUST A FRAME
#
#    # Concatenate everything inside a single file
#    subprocess.call(f'rm -f {bitstream_path}', shell=True)
#    #subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True) # !!!
#    #subprocess.call(f'rm -f {header_path}', shell=True) # !!!
#    # ======================== GOP HEADER ======================== #
#
#    for idx_coding_order in range(video_encoder.coding_structure.get_number_of_frames()):
#        # Retrieve the frame encoder corresponding to the frame
#        frame_encoder, _ = video_encoder.all_frame_encoders.get(str(idx_coding_order))
#
#        frame_bitstream_path = f'{bitstream_path}_{idx_coding_order}'
#        encode_frame(
#            video_encoder,
#            frame_encoder,
#            frame_bitstream_path,
#            idx_coding_order,
#            hls_sig_blksize
#        )
#        subprocess.call(f'cat {frame_bitstream_path} >> {bitstream_path}', shell=True)
#        subprocess.call(f'rm -f {frame_bitstream_path}', shell=True)
#
#    real_rate_byte = os.path.getsize(bitstream_path)
#    # Not very elegant but look at the first frame cool-chic to get the video resolution
#    h, w = video_encoder.all_frame_encoders["0"][0].coolchic_encoder_param.img_size
#    real_rate_bpp = (
#        real_rate_byte * 8 / (h * w * len(video_encoder.coding_structure.frames))
#    )
#    print(f'Real rate        [kBytes]: {real_rate_byte / 1000:9.3f}')
#    print(f'Real rate           [bpp]: {real_rate_bpp :9.3f}')
#
#    elapsed = time.time() - start_time
#    print(f'Encoding time: {elapsed:4.3f} sec')


@torch.no_grad()
def encode_frame(
    frame_encoder: FrameEncoder,
    ref_frame_encoder: FrameEncoder,
    bitstream_path: str,
    display_index: int,
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

    # upsampling has bias parameters, but we do not use them.
    have_bias = { "arm": True,
                  "upsampling": False,
                  "synthesis": True,
                }

    subprocess.call(f'rm -f {bitstream_path}', shell=True)

    # Load the references
    # !!! no references for the moment.

    # Move to pure-int Arms.  Transfer the quantized weights from the fp Arms.
    for cc_name, cc_enc in frame_encoder.coolchic_enc.items():
        arm_fp_param = cc_enc.arm.get_param()
        arm_int = ArmInt(
            cc_enc.param.dim_arm,
            cc_enc.param.n_hidden_layers_arm,
            FIXED_POINT_FRACTIONAL_MULT,
            pure_int=True
        )
        cc_enc.arm = arm_int
        cc_enc.arm.set_param_from_float(arm_fp_param)

    # ================= Encode the MLP into a bitstream file ================ #
    # ac_max_val_nn = get_ac_max_val_nn(frame_encoder)
    ac_max_val_nn = -1 # !!! remove?

    scale_index_nn: DescriptorCoolChic = {} # index with {cc_name}_{module_name}
    q_step_index_nn: DescriptorCoolChic = {} # index with {cc_name}_{module_name}
    n_bytes_nn: DescriptorCoolChic = {} # index with {cc_name}_{module_name}

    for cc_name, cc_enc in frame_encoder.coolchic_enc.items():
        for cur_module_name in cc_enc.modules_to_send:
            # Prepare to store values dedicated to the current modules
            index_name = f'{cc_name}_{cur_module_name}'
            scale_index_nn[index_name] = {}
            q_step_index_nn[index_name] = {}
            n_bytes_nn[index_name] = {}

            module_to_encode = getattr(cc_enc, cur_module_name)

            weights, bias = [], []
            # Retrieve all the weights and biases for the ARM MLP
            q_step_index_nn[index_name]['weight'] = -1
            q_step_index_nn[index_name]['bias'] = -1
            for k, v in module_to_encode.named_parameters():
                assert cur_module_name in ['arm', 'synthesis', 'upsampling'], f'Unknown module name {cur_module_name}. '\
                    'Module name should be in ["arm", "synthesis", "upsampling"].'

                Q_STEPS = POSSIBLE_Q_STEP.get(cur_module_name)

                if "weight" in k:
                    # Find the index of the closest quantization step in the list of
                    # the possible quantization step.
                    cur_possible_q_step = POSSIBLE_Q_STEP.get(cur_module_name).get("weight")
                    cur_q_step = cc_enc.nn_q_step.get(cur_module_name).get("weight")
                    cur_q_step_index = int(
                        torch.argmin((cur_possible_q_step - cur_q_step).abs()).item()
                    )

                    # Store it into q_step_index_nn. It is overwritten for each
                    # loop but it does not matter
                    q_step_index_nn[index_name]["weight"] = cur_q_step_index

                    # Quantize the weight with the actual quantization step and add it
                    # to the list of (quantized) weights
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
                        # No longer relevant without the bi-branch synth
                        # # # Blending -- we get the transformed weight, not the underlying sigmoid parameter.
                        # # # plus: only if >1 branch.
                        # # if cur_module_name == "synthesis" and k.endswith(".parametrizations.weight.original"):
                        # #     if "branch_blender" in k and frame_encoder.coolchic_encoder_param.n_synth_branch == 1:
                        # #         continue # Do not emit unused blender weight.
                        # #     xformed_weights = getattr(module_to_encode, k.replace(".parametrizations.weight.original", "")).weight
                        # #     v = xformed_weights
                        weights.append(
                            torch.round(v / cur_possible_q_step[cur_q_step_index]).flatten()
                        )

                elif "bias" in k and have_bias[cur_module_name]:
                    # Find the index of the closest quantization step in the list of
                    # the Q_STEPS quantization step.
                    cur_possible_q_step = POSSIBLE_Q_STEP.get(cur_module_name).get("bias")
                    cur_q_step = cc_enc.nn_q_step.get(cur_module_name).get("bias")
                    cur_q_step_index = int(
                        torch.argmin((cur_possible_q_step - cur_q_step).abs()).item()
                    )

                    # Store it into q_step_index_nn. It is overwritten for each
                    # loop but it does not matter
                    q_step_index_nn[index_name]["bias"] = cur_q_step_index

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
            if have_bias[cur_module_name]:
                bias = torch.cat(bias).flatten()
            else:
                q_step_index_nn[index_name]['bias'] = 0 # we actually send this in the header.

            # ----------------- Actual entropy coding
            # It happens on cpu
            weights = weights.cpu()
            if have_bias[cur_module_name]:
                bias = bias.cpu()

            cur_bitstream_path = f'{bitstream_path}_{index_name}_weight'

            # either code directly (normal), or search for best (backwards compatible).
            scale_index_weight = cc_enc.nn_expgol_cnt[cur_module_name]['weight']
            if scale_index_weight is None:
                scale_index_weight = -1 # Search for best.
            scale_index_weight = \
                cc_code_wb_bac(cur_bitstream_path,
                            weights.flatten().to(torch.int32).tolist(),
                            scale_index_weight # search for best count if -1
                           )
            scale_index_nn[index_name]['weight'] = scale_index_weight

            n_bytes_nn[index_name]['weight'] = os.path.getsize(cur_bitstream_path)

            if have_bias[cur_module_name]:
                cur_bitstream_path = f'{bitstream_path}_{index_name}_bias'

                # either code directly (normal), or search for best (backwards compatible).
                scale_index_bias = cc_enc.nn_expgol_cnt[cur_module_name]['bias']
                if scale_index_bias is None:
                    scale_index_bias = -1 # Search for best.
                scale_index_bias = \
                cc_code_wb_bac(cur_bitstream_path,
                            bias.flatten().to(torch.int32).tolist(),
                            scale_index_bias # search for best count if -1
                           )
                scale_index_nn[index_name]['bias'] = scale_index_bias

                n_bytes_nn[index_name]['bias'] = os.path.getsize(cur_bitstream_path)
            else:
                scale_index_nn[index_name]['bias'] = 0
                n_bytes_nn[index_name]['bias'] = 0
    # ================= Encode the MLP into a bitstream file ================ #

    # =============== Encode the latent into a bitstream file =============== #
    # To ensure perfect reproducibility between the encoder and the decoder,
    # we load the the different sub-networks from the bitstream here.
    for cc_name, cc_enc in frame_encoder.coolchic_enc.items():
        for cur_module_name in cc_enc.modules_to_send:
            index_name = f'{cc_name}_{cur_module_name}'
            assert cur_module_name in ['arm', 'synthesis', 'upsampling'], f'Unknown module name {cur_module_name}. '\
                'Module name should be in ["arm", "synthesis", "upsampling"].'

            if cur_module_name == 'arm':
                empty_module = ArmInt(
                    cc_enc.param.dim_arm,
                    cc_enc.param.n_hidden_layers_arm,
                    FIXED_POINT_FRACTIONAL_MULT,
                    pure_int = True
                )
            elif cur_module_name == 'synthesis':
                empty_module =  Synthesis(
                    sum(cc_enc.param.n_ft_per_res),
                    cc_enc.param.layers_synthesis
                )
            elif cur_module_name == 'upsampling':
                empty_module = Upsampling(
                        cc_enc.param.ups_k_size,
                        cc_enc.param.ups_preconcat_k_size,
                        # frame_encoder.coolchic_encoder.param.n_ups_kernel,
                        cc_enc.param.latent_n_grids - 1,
                        # frame_encoder.coolchic_encoder.param.n_ups_preconcat_kernel,
                        cc_enc.param.latent_n_grids - 1,
                    )

            Q_STEPS = POSSIBLE_Q_STEP.get(cur_module_name)

            loaded_module = decode_network(
                empty_module,
                DescriptorNN(
                    weight = f'{bitstream_path}_{index_name}_weight',
                    bias = f'{bitstream_path}_{index_name}_bias' if have_bias[cur_module_name] else "",
                ),
                DescriptorNN (
                    weight=Q_STEPS["weight"][q_step_index_nn[index_name]["weight"]],
                    bias=Q_STEPS["bias"][q_step_index_nn[index_name]["bias"]]
                ),
                DescriptorNN (
                    scale_index_nn[index_name]["weight"],
                    bias=(
                        scale_index_nn[index_name]["bias"]
                    )
                    if have_bias[cur_module_name] else 0,
                ),
                ac_max_val_nn
            )
            setattr(cc_enc, cur_module_name, loaded_module)

    # frame_encoder.coolchic_enc.to_device('cpu') # !!! ok? or cycle through sub encoders?
    frame_encoder.set_to_eval()

    # ac_max_val_latent = get_ac_max_val_latent(frame_encoder)
    ac_max_val_latent = -1 # !!! temp -- testing to see if we can remove this.

    # Setting visu to true allows to recover 2D mu, scale and latents
    # We pass in zero-content references.
    res = frame_encoder.coolchic_enc["residue"].latent_grids[0].shape
    res = (res[0], 3, res[2], res[3])
    ref = torch.zeros(res)
    encoder_output = frame_encoder.forward(
        reference_frames=[ref, ref],
        quantizer_noise_type="noise",
        quantizer_type="hardround",
        AC_MAX_VAL=ac_max_val_latent,
        flag_additional_outputs=True,
    )

    # Encode the different 2d latent grids one after the other
    n_bytes_per_latent = {} # index by cc_name to get list.
    hls_blk_sizes = [] # Now per-cc

    for cc_name, cc_enc in frame_encoder.coolchic_enc.items():
        n_bytes_per_latent[cc_name] = []
        best_stream_content = []
        best_stream_sizes = []
        best_stream_names = []
        best_stream_size = None
        best_hls = None
        to_test = [ -32, -16 ]
        if not (hls_sig_blksize in to_test):
            to_test.append(hls_sig_blksize)
        for hls_test in to_test:
            cur_stream_content = []
            cur_stream_size = 0
            cur_stream_sizes = []
            cur_stream_names = []
            ctr_2d_ft = 0
            # Loop on the different resolutions
            for index_lat_resolution in range(cc_enc.param.latent_n_grids):
                current_mu = encoder_output.additional_data.get(f'{cc_name}detailed_mu')[index_lat_resolution]
                current_scale = encoder_output.additional_data.get(f'{cc_name}detailed_scale')[index_lat_resolution]
                current_log_scale = encoder_output.additional_data.get(f'{cc_name}detailed_log_scale')[index_lat_resolution]
                current_y = encoder_output.additional_data.get(f'{cc_name}detailed_sent_latent')[index_lat_resolution]

                c_i, h_i, w_i = current_y.size()[-3:]

                # Nothing to send!
                if c_i == 0:
                    # n_bytes_per_latent[cc_name].append(0)
                    cur_latent_bitstream = get_sub_bitstream_path(f'{bitstream_path}_{cc_name}', ctr_2d_ft)
                    cur_stream_sizes.append(0)
                    cur_stream_content.append(None)
                    cur_stream_names.append(cur_latent_bitstream)
                    # Still create an empty file for coherence
                    subprocess.call(f'touch {cur_latent_bitstream}', shell=True)
                    ctr_2d_ft += 1
                    continue

                # Loop on the different 2D grids composing one resolutions
                for index_lat_feature in range(c_i):
                    y_this_ft = current_y[:, index_lat_feature, :, :].flatten().cpu()
                    mu_this_ft = current_mu[:, index_lat_feature, :, :].flatten().cpu()
                    log_scale_this_ft = current_log_scale[:, index_lat_feature, :, :].flatten().cpu()

                    if y_this_ft.abs().max() == 0:
                        # n_bytes_per_latent[cc_name].append(0)
                        cur_latent_bitstream = get_sub_bitstream_path(f'{bitstream_path}_{cc_name}', ctr_2d_ft)
                        cur_stream_sizes.append(0)
                        cur_stream_content.append(None)
                        cur_stream_names.append(cur_latent_bitstream)
                        # Still create an empty file for coherence
                        subprocess.call(f'touch {cur_latent_bitstream}', shell=True)
                        ctr_2d_ft += 1
                        continue

                    cur_latent_bitstream = get_sub_bitstream_path(f'{bitstream_path}_{cc_name}', ctr_2d_ft)
                    cc_code_latent_layer_bac(
                        cur_latent_bitstream,
                        y_this_ft.flatten().to(torch.int32).tolist(),
                        (mu_this_ft*FIXED_POINT_FRACTIONAL_MULT).round().flatten().to(torch.int32).tolist(),
                        (log_scale_this_ft*FIXED_POINT_FRACTIONAL_MULT).round().flatten().to(torch.int32).tolist(),
                        h_i, w_i,
                        hls_test,
                    )

                    with open(cur_latent_bitstream, "rb") as f:
                        cur_stream_content.append(f.read())
                    cur_stream_sizes.append(os.path.getsize(cur_latent_bitstream))
                    cur_stream_names.append(cur_latent_bitstream)
                    cur_stream_size += os.path.getsize(cur_latent_bitstream)
                    # n_bytes_per_latent[cc_name].append(os.path.getsize(cur_latent_bitstream))
                    ctr_2d_ft += 1

            if best_stream_size is None \
                or cur_stream_size < best_stream_size:
                print(cur_stream_size, "<", best_stream_size, "with", hls_test)
                best_stream_size = cur_stream_size
                best_stream_sizes = cur_stream_sizes
                best_stream_content = cur_stream_content
                best_stream_names = cur_stream_names
                best_hls = hls_test
            else:
                print("skipping", cur_stream_size, "with", hls_test)

        # Use the best.
        hls_blk_sizes.append(best_hls)
        n_bytes_per_latent[cc_name] = best_stream_sizes
        print("using", best_hls, "perlatent:", n_bytes_per_latent[cc_name])
        for idx in range(len(best_stream_names)):
            if best_stream_content[idx] is None:
                subprocess.call(f'rm -f {best_stream_names[idx]}', shell=True)
                subprocess.call(f'touch {best_stream_names[idx]}', shell=True)
            else:
                with open(best_stream_names[idx], "wb") as f:
                    f.write(best_stream_content[idx])

    # Write a gop header for this frame
    if display_index == 0:
        gop_header_path = f'{bitstream_path}_gopheader'
        write_gop_header(
            gop_header_path,
            frame_encoder.coolchic_enc['residue'].param.img_size[-2],
            frame_encoder.coolchic_enc['residue'].param.img_size[-1],
            frame_encoder.frame_data_type,
            frame_encoder.bitdepth)

    # Write the frame header
    header_path = f'{bitstream_path}_header'
    write_frame_header(
        frame_encoder,
        ref_frame_encoder,
        header_path,
        display_index,
        n_bytes_per_latent,
        q_step_index_nn,
        scale_index_nn,
        n_bytes_nn,
        hls_blk_sizes,
    )

    # Concatenate everything inside a single file
    subprocess.call(f'rm -f {bitstream_path}', shell=True)
    if display_index == 0:
        subprocess.call(f'cat {gop_header_path} {header_path} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {gop_header_path} {header_path}', shell=True)
    else:
        subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {header_path}', shell=True)

    for cc_name, cc_enc in frame_encoder.coolchic_enc.items():
        latents_zero = cc_latents_zero(cc_enc, n_bytes_per_latent[cc_name])
        print("latents_zero", latents_zero)
        for cur_module_name in ['arm', 'upsampling', 'synthesis']:
            for parameter_type in ['weight', 'bias']:
                cur_bitstream = f'{bitstream_path}_{cc_name}_{cur_module_name}_{parameter_type}'
                if os.path.exists(cur_bitstream):
                    if latents_zero and cur_module_name == "upsampling":
                        pass
                    else:
                        subprocess.call(f'cat {cur_bitstream} >> {bitstream_path}', shell=True)
                    subprocess.call(f'rm -f {cur_bitstream}', shell=True)

        ctr_2d_ft = 0
        for index_lat_resolution in range(cc_enc.param.latent_n_grids):

            # No feature: still increment the counter and remove the temporary bitstream file
            if cc_enc.latent_grids[index_lat_resolution].size()[1] == 0:
                cur_latent_bitstream = get_sub_bitstream_path(f'{bitstream_path}_{cc_name}', ctr_2d_ft)
                subprocess.call(f'rm -f {cur_latent_bitstream}', shell=True)
                ctr_2d_ft += 1

            for index_lat_feature in range(cc_enc.latent_grids[index_lat_resolution].size()[1]):
                cur_latent_bitstream = get_sub_bitstream_path(f'{bitstream_path}_{cc_name}', ctr_2d_ft)
                if latents_zero:
                    pass
                else:
                    subprocess.call(f'cat {cur_latent_bitstream} >> {bitstream_path}', shell=True)
                subprocess.call(f'rm -f {cur_latent_bitstream}', shell=True)
                ctr_2d_ft += 1

    # Encoding's done, we no longer need deterministic algorithms
    torch.use_deterministic_algorithms(False)


def get_sub_bitstream_path(
    root_bitstream_path: str,
    counter_2d_latent: int,
) -> str:
    """Return the complete path of the sub-bistream corresponding to the
    <counter_2d_latent>-th 2D feature maps. This is use due to the fact that
    even 3D features are coded as independent 2D features.

    Args:
        root_bitstream_path (str): Root name of the bitstream
        counter_2d_latent (int): Index of 2D features. Let us suppose that we have
            two features maps, one is [1, 1, H, W] and the other is [1, 2, H/2, W/2].
            The counter_2d_latent will be used to iterate on the **3** 2d features
            (one for the highest resolution, two for the smallest resolution).

    Returns:
        str: Complete bitstream path
    """
    s = f'{root_bitstream_path}_{counter_2d_latent}'
    return s
