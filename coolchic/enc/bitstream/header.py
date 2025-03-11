# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


"""
Bitstream structure
-------------------

Limitations:
    - MLPs hidden layers must be of constant size e.g. 24,24 or 16,16
    - One bitstream = one gop i.e. something starting with an intra frame

Header for the GOP (one by video):
----------------------------------

    ? ========================= GOP HEADER ========================= ?
    [Number of bytes used for the header]           2 bytes
    [Image height]                                  2 bytes
    [Image width]                                   2 bytes

    ? Everything here is detailed in encoder/utils/coding_structure.py
    [FRAME_DATA_TYPE, BITDEPTH]                     1 byte (Everything condensed in one byte)
    [Intra Period]                                  1 byte (From 0 to 255)
    [P-period]                                      1 byte (From 0 to 255)
    ? ========================= GOP HEADER ========================= ?


Header for the frame (one by frame):
------------------------------------

    ? ======================== FRAME HEADER ======================== ?
    [Number of bytes used for the header]           2 bytes
    [Display index of the frame]                    1 byte (From 0. to 255.) Not absolutely necessary!
    [ARM Hidden layer dim, num_hidden_layer]        1 byte  (Everything condensed in one byte)
    [Upsampling kernel size, static kernel flag]    1 byte  (Everything condensed in one byte)
    [Number of layer Synthesis]                     1 byte  (e.g. 2 if archi='24,24')
    [Synthesis layer 0 description ]                3 bytes  (Contains out_ft / k_size / layer_type / non-linearity)
                        ...
    [Synthesis layer N - 1 description ]            3 bytes  (Contains out_ft / k_size / layer_type / non-linearity)
    [Flow gain]                                     1 byte  (From 0. to 255.)


    [Index of quantization step weight ARM]         1 bytes (From 0 to 255)
    [Index of quantization step bias ARM]           1 bytes (From 0 to 255)
    [Index of quantization step weight Upsampling]  1 bytes (From 0 to 255)
    [Index of quantization step bias Upsampling]    1 bytes (From 0 to 255)
    [Index of quantization step weight Synthesis]   1 bytes (From 0 to 255)
    [Index of quantization step bias Synthesis]     1 bytes (From 0 to 255)

    [Index of scale entropy coding weight ARM]      1 bytes (From 0 to 2 ** 8 - 1)
    [Index of scale entropy coding bias ARM]        1 bytes (From 0 to 2 ** 8 - 1)
    [Idx of scale entropy coding weight Upsampling] 1 bytes (From 0 to 2 ** 8 - 1)
    [Idx of scale entropy coding bias Upsampling]   1 bytes (From 0 to 2 ** 8 - 1)
    [Idx of scale entropy coding weight Synthesis]  1 bytes (From 0 to 2 ** 8 - 1)
    [Idx of scale entropy coding bias Synthesis]    1 bytes (From 0 to 2 ** 8 - 1)

    [Number of bytes used for weight ARM]           2 bytes (less than 65535 bytes)
    [Number of bytes used for bias ARM]             2 bytes (less than 65535 bytes)
    [Number of bytes used for weight Upsampling]    2 bytes (less than 65535 bytes)
    [Number of bytes used for bias Upsampling]      2 bytes (less than 65535 bytes)
    [Number of bytes used for weight Synthesis]     2 bytes (less than 65535 bytes)
    [Number of bytes used for bias Synthesis]       2 bytes (less than 65535 bytes)

    ! Here, we split the 3D latent grids into several 2D grids. For instance,
    ! if we have three latent resolutions:
    !       [1, H, W] ; [3, H / 2, W / 2], [0, H / 4, W / 4] and [2, H / 8, W / 8]
    ! We have 1 + 3 + 0 + 2 = 6 2D latent grids but 4 latent resolutions
    ! Resolution with 0 feature are still displayed in "Number of feature maps for latent X"
    ! and "Number of bytes used for latent X" with value set to 0
    [Number of latent resolution]                   1 byte
    [Number of **2D** latent grids]                 1 byte
    [Number of feature maps for latent 0]           1 byte
    [Number of feature maps for latent 1]           1 byte
                        ...
    [Number of feature maps for latent N - 1]       1 byte
    [Number of bytes used for latent grid 0]        3 bytes (less than 16777215 bytes)
    [Number of bytes used for latent grid 1]        3 bytes (less than 16777215 bytes)
                        ...
    [Number of bytes used for latent grid N - 1]    3 bytes (less than 16777215 bytes)
    ? ======================== FRAME HEADER ======================== ?

"""

import os
from typing import List, Tuple, TypedDict

from enc.component.core.synthesis import Synthesis
from enc.component.frame import FrameEncoder
from enc.component.types import DescriptorCoolChic
# from enc.component.video import VideoEncoder


# Quick & dirty copy and paste from utils.coding_structure
_FRAME_DATA_TYPE = ["rgb", "yuv420", "yuv444"]
_POSSIBLE_BITDEPTH = [8, 9, 10, 11, 12, 13, 14, 15, 16]
_POSSIBLE_SYNTHESIS_MODE = [k for k in Synthesis.possible_mode]
_POSSIBLE_SYNTHESIS_NON_LINEARITY = [k for k in Synthesis.possible_non_linearity]


#class GopHeader(TypedDict):
#    n_bytes_header: int  # Number of bytes for the header
#    img_size: Tuple[int, int]  # Format: (height, width)
#    frame_data_type: FRAME_DATA_TYPE  # RGB, YUV 4:2:0, YUV 4:4:4
#    bitdepth: POSSIBLE_BITDEPTH  # 8 through 16
#    intra_period: int  # See coding_structure.py
#    p_period: int  # See coding_structure.py


# temporary
def write_gop_header(header_path: str, image_height: int, image_width: int, frame_data_type, output_bitdepth: int):
    """Write a frame header to a a file located at <header_path>.
    The structure of the header is described above.

    Args:
        video_encoder (VideoEncoder): Model from which info located in the header will be retrieved.
        header_path (str): Path of the file where the header is written.
    """

    n_bytes_header = 0
    n_bytes_header += 1  # Number of bytes header
    n_bytes_header += 2  # Image height
    n_bytes_header += 2  # Image width
    n_bytes_header += 1  # frame data type << 4 | (output_bitdepth-8)

    byte_to_write = b""
    byte_to_write += n_bytes_header.to_bytes(1, byteorder="big", signed=False)

    byte_to_write += image_height.to_bytes(2, byteorder="big", signed=False)
    byte_to_write += image_width.to_bytes(2, byteorder="big", signed=False)

    send_output_bitdepth = output_bitdepth-8
    if send_output_bitdepth < 0 or send_output_bitdepth >= 16:
        print("sending bitdepth {output_bitdepth} outside expected limits")
        exit(1)

    byte_to_write += ((_FRAME_DATA_TYPE.index(frame_data_type)<<4) | send_output_bitdepth) \
                        .to_bytes(1, byteorder="big", signed=False)

    with open(header_path, "wb") as fout:
        fout.write(byte_to_write)

    if n_bytes_header != os.path.getsize(header_path):
        print("Invalid number of bytes in header!")
        print("expected", n_bytes_header)
        print("got", os.path.getsize(header_path))
        exit(1)

#def write_gop_header(video_encoder: VideoEncoder, header_path: str):
#    """Write a frame header to a a file located at <header_path>.
#    The structure of the header is described above.
#
#    Args:
#        video_encoder (VideoEncoder): Model from which info located in the header will be retrieved.
#        header_path (str): Path of the file where the header is written.
#    """
#
#    n_bytes_header = 0
#    n_bytes_header += 2  # Number of bytes header
#    n_bytes_header += 2  # Image height
#    n_bytes_header += 2  # Image width
#    n_bytes_header += 1  # GOP type, frame data type, bitdepth
#    n_bytes_header += 1  # intra period
#    n_bytes_header += 1  # p period
#
#    byte_to_write = b""
#    byte_to_write += n_bytes_header.to_bytes(2, byteorder="big", signed=False)
#
#    # Get these to access to the img_size in frame_encoder.coolchic_encoder_param
#    intra_frame_encoder, _ = video_encoder.all_frame_encoders["0"]
#
#    byte_to_write += intra_frame_encoder.coolchic_encoder_param.img_size[-2].to_bytes(
#        2, byteorder="big", signed=False
#    )
#    byte_to_write += intra_frame_encoder.coolchic_encoder_param.img_size[-1].to_bytes(
#        2, byteorder="big", signed=False
#    )
#
#    byte_to_write += (
#        # Last 4 bits are for the bitdepth
#        _POSSIBLE_BITDEPTH.index(intra_frame_encoder.bitdepth) * 2**4
#        # The first 4 bits are for the frame data type
#        + _FRAME_DATA_TYPE.index(intra_frame_encoder.frame_data_type)
#    ).to_bytes(1, byteorder="big", signed=False)
#
#    byte_to_write += video_encoder.coding_structure.intra_period.to_bytes(
#        1, byteorder="big", signed=False
#    )
#    byte_to_write += video_encoder.coding_structure.p_period.to_bytes(
#        1, byteorder="big", signed=False
#    )
#
#    with open(header_path, "wb") as fout:
#        fout.write(byte_to_write)
#
#    if n_bytes_header != os.path.getsize(header_path):
#        print("Invalid number of bytes in header!")
#        print("expected", n_bytes_header)
#        print("got", os.path.getsize(header_path))
#        exit(1)
#

#def read_gop_header(bitstream_path: str) -> GopHeader:
#    """Read the first few bytes of a bitstream file located at
#    <bitstream_path> and parse the different information.
#
#    Args:
#        bitstream_path (str): Path where the bitstream is located.
#
#    Returns:
#        GopHeader: The parsed info from the bitstream.
#    """
#    with open(bitstream_path, "rb") as fin:
#        bitstream = fin.read()
#
#    ptr = 0
#    n_bytes_header = int.from_bytes(
#        bitstream[ptr : ptr + 2], byteorder="big", signed=False
#    )
#    ptr += 2
#
#    img_height = int.from_bytes(bitstream[ptr : ptr + 2], byteorder="big", signed=False)
#    ptr += 2
#    img_width = int.from_bytes(bitstream[ptr : ptr + 2], byteorder="big", signed=False)
#    ptr += 2
#
#    raw_value = int.from_bytes(bitstream[ptr : ptr + 1], byteorder="big", signed=False)
#    ptr += 1
#    bitdepth = _POSSIBLE_BITDEPTH[raw_value // 2**4]
#    frame_data_type = _FRAME_DATA_TYPE[(raw_value % 2**4)]
#
#    intra_period = int.from_bytes(
#        bitstream[ptr : ptr + 1], byteorder="big", signed=False
#    )
#    ptr += 1
#    p_period = int.from_bytes(bitstream[ptr : ptr + 1], byteorder="big", signed=False)
#    ptr += 1
#
#    header_info: GopHeader = {
#        "n_bytes_header": n_bytes_header,
#        "img_size": (img_height, img_width),
#        "frame_data_type": frame_data_type,
#        "bitdepth": bitdepth,
#        "intra_period": intra_period,
#        "p_period": p_period,
#    }
#
#    print("\nContent of the GOP header:")
#    print("---------------------------")
#    for k, v in header_info.items():
#        print(f"{k:>20}: {v}")
#    print("         ---------------------")
#
#    return header_info


class FrameHeader(TypedDict):
    """Define the dictionary containing the header information as a type."""

    n_bytes_header: int  # Number of bytes for the header
    display_index: int  # Display index of the frame
    latent_n_resolutions: int  # Number of different latent resolutions
    latent_n_2d_grid: int  # Number of 2D latent grids
    n_ft_per_latent: List[int]  # Number of feature maps for each latent resolutions
    n_bytes_per_latent: List[int]  # Number of bytes for each 2D latent
    img_size: Tuple[int, int]  # Format: (height, width)
    dim_arm: (
        int  # Number of context pixels AND dimension of the hidden layers of the ARM
    )
    n_hidden_layers_arm: int  # Number of hidden layers in the ARM. 0 for a linear ARM
    upsampling_kernel_size: int  # Upsampling model kernel size
    static_upsampling_kernel: (
        bool  # Whether the upsampling kernel is learned or kept static (bicubic)
    )
    layers_synthesis: List[str]  # Dimension of each hidden layer in the Synthesis
    q_step_index_nn: DescriptorCoolChic  # Index of the quantization step used for the weight & bias of the NNs
    scale_index_nn: DescriptorCoolChic  # Index of the scale used for entropy code weight & bias of the NNs
    n_bytes_nn: DescriptorCoolChic  # Number of bytes for weight and bias of the NNs
    ac_max_val_nn: (
        int  # The range coder AC_MAX_VAL parameters for entropy coding the NNs
    )
    ac_max_val_latent: (
        int  # The range coder AC_MAX_VAL parameters for entropy coding the latents
    )
    hls_sig_blksize: int                # zero-significants-in-block block detection size
    image_type: str  # Image type from the list possible_image_type defined above
    flow_gain: int  # Multiply the optical flow coming from cool chic

def code_frame_type(frame_type):
    if frame_type == 'I':
        return 0
    if frame_type == 'P':
        return 1
    if frame_type == 'B':
        return 2
    print("internal error frame_type", frame_type)
    exit(1)

def utf_code(val, signed = False):
    if val < 0 and not signed:
        print("internal error: utf_code with -ve value", val)
        exit(1)

    # With a signed value, the +ve value is transmitted doubled with the sign bit as the low bit.

    signbit = 0
    if signed:
        signbit = 1 if val < 0 else 0
        val = -val if val < 0 else val
        val <<= 1
        val |= signbit

    nbytes = 1
    # 7 bits per byte output.  How many bytes to code val?

    while (1<<(7*nbytes)) <= val:
        nbytes += 1

    byte_to_write = b""
    while nbytes > 0:
        byte_to_write += ((1<<7)|(val>>(nbytes*7)&0x7f)).to_bytes(1, byteorder="big", signed=False)
        nbytes -= 1

    # High bit not set on last byte.
    byte_to_write += ((0<<7)|(val&((1<<7)-1))).to_bytes(1, byteorder="big", signed=False)

    return byte_to_write

# ref_frame_encoder: optional frame encoder from which to get a cc.
# cc_idx: the index we are after (0 or 1)
# Returns None for nothing appropriate.
def get_ref_cc(ref_frame_encoder, cc_idx):
    if ref_frame_encoder is None:
        return None

    cc_search_idx = 0
    for cc_name, cc_enc in ref_frame_encoder.coolchic_enc.items():
        result = cc_enc
        if cc_search_idx == cc_idx:
            break
        cc_search_idx += 1

    print("get_ref_cc:", cc_idx, "returns", cc_search_idx)
    return result

# Returns a bit mask of what topologies are equal:
# ARM bit 0
# UPS bit 1
# SYN bit 2
# LAT bit 3
def cc_topologies_equal(cc_enc_1, cc_enc_2):
    if cc_enc_2 is None:
        return 0

    result = 0

    #print(f"topology wanted: arm {cc_enc_1.param.dim_arm} {cc_enc_1.param.n_hidden_layers_arm} \
    #        ups {cc_enc_1.upsampling.n_ups_kernel} {cc_enc_1.upsampling.n_ups_preconcat_kernel} {cc_enc_1.param.ups_k_size} {cc_enc_1.param.ups_preconcat_k_size} \
    #        syn {cc_enc_1.param.layers_synthesis}\
    #        lat {cc_enc_1.param.latent_n_grids} {cc_enc_1.param.n_ft_per_res}")
    if cc_enc_1.param.dim_arm == cc_enc_2.param.dim_arm \
        and cc_enc_1.param.n_hidden_layers_arm == cc_enc_2.param.n_hidden_layers_arm:
        result |= (1<<0)
    if cc_enc_1.upsampling.n_ups_kernel == cc_enc_2.upsampling.n_ups_kernel \
        and cc_enc_1.upsampling.n_ups_preconcat_kernel == cc_enc_2.upsampling.n_ups_preconcat_kernel \
        and cc_enc_1.param.ups_k_size == cc_enc_2.param.ups_k_size \
        and cc_enc_1.param.ups_preconcat_k_size == cc_enc_2.param.ups_preconcat_k_size:
        result |= (1<<1)
    if cc_enc_1.param.layers_synthesis == cc_enc_2.param.layers_synthesis:
        result |= (1<<2)
    if cc_enc_1.param.latent_n_grids == cc_enc_2.param.latent_n_grids \
        and cc_enc_1.param.n_ft_per_res == cc_enc_2.param.n_ft_per_res: # list comparison
        result |= (1<<3)

    #print("returning 0x%x"%(result))
    return result

def code_cc_topology(cc_enc, topologies_equal_bits):
    byte_to_write = b""

    # Since the dimension of the hidden layer and of the context is always a multiple of
    # 8, we can spare 3 bits by dividing it by 8
    assert cc_enc.param.dim_arm // 8 < 2**4, (
        f"Number of context pixels"
        f" and dimension of the hidden layer for the arm must be inferior to {2 ** 4 * 8}. Found"
        f" {cc_enc.dim_arm} in encoder {cc_name}"
    )

    assert cc_enc.param.n_hidden_layers_arm < 2**4, (
        f"Number of hidden layers"
        f" for the ARM should be inferior to {2 ** 4}. Found "
        f"{cc_enc.n_hidden_layers_arm} in {cc_name}"
    )

    if (topologies_equal_bits&(1<<0)) == 0:
        byte_to_write += (
            ((cc_enc.param.dim_arm // 8) << 4) \
            | cc_enc.param.n_hidden_layers_arm
        ).to_bytes(1, byteorder="big", signed=False)

    if (topologies_equal_bits&(1<<1)) == 0:
        byte_to_write += (
            ((cc_enc.upsampling.n_ups_kernel)<<4)|(cc_enc.param.ups_k_size)
        ).to_bytes(1, byteorder="big", signed=False)
        byte_to_write += (
            ((cc_enc.upsampling.n_ups_preconcat_kernel)<<4)|(cc_enc.param.ups_preconcat_k_size)
        ).to_bytes(1, byteorder="big", signed=False)

    if (topologies_equal_bits&(1<<2)) == 0:
        byte_to_write += len(
            cc_enc.param.layers_synthesis
        ).to_bytes(1, byteorder="big", signed=False)

        # If no hidden layers in the Synthesis, cc_enc.param.layers_synthesis is an empty list. So write 0
        for layer_spec in cc_enc.param.layers_synthesis:
            out_ft, k_size, mode, non_linearity = layer_spec.split("-")
            byte_to_write += int(out_ft).to_bytes(1, byteorder="big", signed=False)
            byte_to_write += int(k_size).to_bytes(1, byteorder="big", signed=False)
            byte_to_write += (
                _POSSIBLE_SYNTHESIS_MODE.index(mode) * 16
                + _POSSIBLE_SYNTHESIS_NON_LINEARITY.index(non_linearity)
            ).to_bytes(1, byteorder="big", signed=False)

    if (topologies_equal_bits&(1<<3)) == 0:
        byte_to_write += cc_enc.param.latent_n_grids.to_bytes(
            1, byteorder="big", signed=False
        )

        for i, latent_i in enumerate(cc_enc.latent_grids):
            n_ft_i = latent_i.size()[1]
            if n_ft_i > 2**8 - 1:
                print(f"Number of feature maps for latent {i} is too big in {cc_name}!")
                print(f"Found {n_ft_i}, should be smaller than {2 ** 8 - 1}")
                print("Exiting!")
                return
            byte_to_write += n_ft_i.to_bytes(1, byteorder="big", signed=False)

    return byte_to_write

# returns 0 if latents contain non-zero, else 1
def cc_latents_zero(cc_enc, n_bytes_per_latent):
    # We look at 0-sized coding result which is emitted if all latents were zero.
    for l in n_bytes_per_latent:
        if l != 0:
            return 0
    return 1

def write_frame_header(
    frame_encoder: FrameEncoder,
    ref_frame_encoder: FrameEncoder,
    header_path: str,
    display_index: int,
    n_bytes_per_latent: List[int],
    q_step_index_nn: DescriptorCoolChic,
    scale_index_nn: DescriptorCoolChic,
    n_bytes_nn: DescriptorCoolChic,
    hls_sig_blksize: List[int],
):
    """Write a frame header to a a file located at <header_path>.
    The structure of the header is described above.

    Args:
        frame_encoder (FrameEncoder): Model from which info located in the header will be retrieved.
        header_path (str): Path of the file where the header is written.
        n_bytes_per_latent (List[int]): Indicates the number of bytes for each 2D
            latent grid.
        q_step_index_nn (DescriptorCoolChic): Dictionary containing the index of the
            quantization step for the weight and bias of each network.
        scale_index_nn (DescriptorCoolChic): Dictionary containing the index of the
            scale parameter used during the entropy coding of the weight and bias
            of each network.
        n_bytes_nn (DescriptorCoolChic): Dictionary containing the number of bytes
            used for the weights and biases of each network
    """


    print(f'FRAME TYPE {frame_encoder.frame_type}')

    # UTF:  n_header_bytes
    # UTF:  display_index

    byte_to_write = b""
    byte_to_write += utf_code(display_index)

    frame_type_bits = code_frame_type(frame_encoder.frame_type)

    # BITS: (hi to lo)
    # BITS: lat_all_zero:   2 bits I->1+0, P|B->2
    # BITS: frame_type: 2 bits (I 00, P 01, B 10)

    cc_eq_bits = [0, 0]
    cc_latz_bits = [0, 0]

    cc_idx = 0
    for cc_name, cc_enc in frame_encoder.coolchic_enc.items():
        cc_eq_bits[cc_idx] = cc_topologies_equal(cc_enc, get_ref_cc(ref_frame_encoder, cc_idx))
        cc_latz_bits[cc_idx] = cc_latents_zero(cc_enc, n_bytes_per_latent[cc_name])
        cc_idx += 1

    if frame_encoder.frame_type == "I":
        # frame-type(2), latz0(1).
        byte_to_write += ((frame_type_bits)|(cc_latz_bits[0]<<2)).to_bytes(1, byteorder="big", signed=False)
    else:
        # frame_type(2), latz0(1), cceq0(4)
        # latz1(1), cceq1(4)
        byte_to_write += ((frame_type_bits)|(cc_latz_bits[0]<<2)|(cc_eq_bits[0]<<3)).to_bytes(1, byteorder="big", signed=False)
        byte_to_write += ((cc_latz_bits[1])|(cc_eq_bits[1]<<1)).to_bytes(1, byteorder="big", signed=False)

    print("topology-equal-bits: 0x%x 0x%x"%(cc_eq_bits[0], cc_eq_bits[1]))
    print("latents-zero-bits:", cc_latz_bits)

    # global flows
    if frame_encoder.frame_type == "P" or frame_encoder.frame_type == "B":
        byte_to_write += utf_code(int(frame_encoder.global_flow_1[0,0,0,0]), signed=True)
        byte_to_write += utf_code(int(frame_encoder.global_flow_1[0,1,0,0]), signed=True)
        print("global_flow_1", frame_encoder.global_flow_1)
        if frame_encoder.frame_type == "B":
            byte_to_write += utf_code(int(frame_encoder.global_flow_2[0,0,0,0]), signed=True)
            byte_to_write += utf_code(int(frame_encoder.global_flow_2[0,1,0,0]), signed=True)
            print("global_flow_2", frame_encoder.global_flow_2)

    # Write, if necessary, CC topology, followed by quantization levels and later CABAC coded substream lengths.
    cc_idx = 0
    for cc_name, cc_enc in frame_encoder.coolchic_enc.items():
        byte_to_write += code_cc_topology(cc_enc, cc_eq_bits[cc_idx])

        byte_to_write += hls_sig_blksize[cc_idx].to_bytes(1, byteorder='big', signed=True)

        latents_zero = cc_latents_zero(cc_enc, n_bytes_per_latent[cc_name])
        if latents_zero: # no upsampling if latents are all zero.
            modules = ["arm", "synthesis"]
        else:
            modules = ["arm", "upsampling", "synthesis"]

        for nn_name in modules:
            index_name = f'{cc_name}_{nn_name}'
            for nn_param in ["weight", "bias"]:
                cur_q_step_index = q_step_index_nn.get(index_name).get(nn_param)
                byte_to_write += cur_q_step_index.to_bytes(
                    1, byteorder="big", signed=False
                )

        for nn_name in modules:
            index_name = f'{cc_name}_{nn_name}'
            for nn_param in ["weight", "bias"]:
                cur_scale_index = scale_index_nn.get(index_name).get(nn_param)
                byte_to_write += cur_scale_index.to_bytes(1, byteorder="big", signed=False)

        for nn_name in modules:
            index_name = f'{cc_name}_{nn_name}'
            for nn_param in ["weight", "bias"]:
                cur_n_bytes = n_bytes_nn.get(index_name).get(nn_param)
                byte_to_write += utf_code(cur_n_bytes)

        if not latents_zero:
            for i, v in enumerate(n_bytes_per_latent[cc_name]):
                byte_to_write += utf_code(v)

        cc_idx += 1

    # Get and write the header length.
    first_header_len_utf = utf_code(len(byte_to_write))
    header_len_utf = utf_code(len(byte_to_write)+len(first_header_len_utf))
    if header_len_utf > first_header_len_utf:
        header_len_utf = utf_code(len(byte_to_write)+len(header_len_utf))
    else:
        header_len_utf = first_header_len_utf

    byte_to_write = header_len_utf+byte_to_write

    with open(header_path, "wb") as fout:
        fout.write(byte_to_write)
