# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
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
from enc.component.video import VideoEncoder
from enc.utils.codingstructure import FRAME_DATA_TYPE, POSSIBLE_BITDEPTH, Frame
from enc.utils.misc import MAX_AC_MAX_VAL, DescriptorCoolChic


# Quick & dirty copy and paste from utils.coding_structure
_FRAME_DATA_TYPE = ["rgb", "yuv420", "yuv444"]
_POSSIBLE_BITDEPTH = [8, 9, 10, 11, 12, 13, 14, 15, 16]
_POSSIBLE_SYNTHESIS_MODE = [k for k in Synthesis.possible_mode]
_POSSIBLE_SYNTHESIS_NON_LINEARITY = [k for k in Synthesis.possible_non_linearity]


class GopHeader(TypedDict):
    n_bytes_header: int  # Number of bytes for the header
    img_size: Tuple[int, int]  # Format: (height, width)
    frame_data_type: FRAME_DATA_TYPE  # RGB, YUV 4:2:0, YUV 4:4:4
    bitdepth: POSSIBLE_BITDEPTH  # 8 through 16
    intra_period: int  # See coding_structure.py
    p_period: int  # See coding_structure.py


def write_gop_header(video_encoder: VideoEncoder, header_path: str):
    """Write a frame header to a a file located at <header_path>.
    The structure of the header is described above.

    Args:
        video_encoder (VideoEncoder): Model from which info located in the header will be retrieved.
        header_path (str): Path of the file where the header is written.
    """

    n_bytes_header = 0
    n_bytes_header += 2  # Number of bytes header
    n_bytes_header += 2  # Image height
    n_bytes_header += 2  # Image width
    n_bytes_header += 1  # GOP type, frame data type, bitdepth
    n_bytes_header += 1  # intra period
    n_bytes_header += 1  # p period

    byte_to_write = b""
    byte_to_write += n_bytes_header.to_bytes(2, byteorder="big", signed=False)

    # Get these to access to the img_size in frame_encoder.coolchic_encoder_param
    intra_frame_encoder, _ = video_encoder.all_frame_encoders["0"]

    byte_to_write += intra_frame_encoder.coolchic_encoder_param.img_size[-2].to_bytes(
        2, byteorder="big", signed=False
    )
    byte_to_write += intra_frame_encoder.coolchic_encoder_param.img_size[-1].to_bytes(
        2, byteorder="big", signed=False
    )

    byte_to_write += (
        # Last 4 bits are for the bitdepth
        _POSSIBLE_BITDEPTH.index(intra_frame_encoder.bitdepth) * 2**4
        # The first 4 bits are for the frame data type
        + _FRAME_DATA_TYPE.index(intra_frame_encoder.frame_data_type)
    ).to_bytes(1, byteorder="big", signed=False)

    byte_to_write += video_encoder.coding_structure.intra_period.to_bytes(
        1, byteorder="big", signed=False
    )
    byte_to_write += video_encoder.coding_structure.p_period.to_bytes(
        1, byteorder="big", signed=False
    )

    with open(header_path, "wb") as fout:
        fout.write(byte_to_write)

    if n_bytes_header != os.path.getsize(header_path):
        print("Invalid number of bytes in header!")
        print("expected", n_bytes_header)
        print("got", os.path.getsize(header_path))
        exit(1)


def read_gop_header(bitstream_path: str) -> GopHeader:
    """Read the first few bytes of a bitstream file located at
    <bitstream_path> and parse the different information.

    Args:
        bitstream_path (str): Path where the bitstream is located.

    Returns:
        GopHeader: The parsed info from the bitstream.
    """
    with open(bitstream_path, "rb") as fin:
        bitstream = fin.read()

    ptr = 0
    n_bytes_header = int.from_bytes(
        bitstream[ptr : ptr + 2], byteorder="big", signed=False
    )
    ptr += 2

    img_height = int.from_bytes(bitstream[ptr : ptr + 2], byteorder="big", signed=False)
    ptr += 2
    img_width = int.from_bytes(bitstream[ptr : ptr + 2], byteorder="big", signed=False)
    ptr += 2

    raw_value = int.from_bytes(bitstream[ptr : ptr + 1], byteorder="big", signed=False)
    ptr += 1
    bitdepth = _POSSIBLE_BITDEPTH[raw_value // 2**4]
    frame_data_type = _FRAME_DATA_TYPE[(raw_value % 2**4)]

    intra_period = int.from_bytes(
        bitstream[ptr : ptr + 1], byteorder="big", signed=False
    )
    ptr += 1
    p_period = int.from_bytes(bitstream[ptr : ptr + 1], byteorder="big", signed=False)
    ptr += 1

    header_info: GopHeader = {
        "n_bytes_header": n_bytes_header,
        "img_size": (img_height, img_width),
        "frame_data_type": frame_data_type,
        "bitdepth": bitdepth,
        "intra_period": intra_period,
        "p_period": p_period,
    }

    print("\nContent of the GOP header:")
    print("---------------------------")
    for k, v in header_info.items():
        print(f"{k:>20}: {v}")
    print("         ---------------------")

    return header_info


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


def write_frame_header(
    frame_encoder: FrameEncoder,
    frame: Frame,
    header_path: str,
    n_bytes_per_latent: List[int],
    q_step_index_nn: DescriptorCoolChic,
    scale_index_nn: DescriptorCoolChic,
    n_bytes_nn: DescriptorCoolChic,
    ac_max_val_nn: int,
    ac_max_val_latent: int,
    hls_sig_blksize: int,
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
        ac_max_val_nn (int): The range coder AC_MAX_VAL parameters for entropy coding the NNs
        ac_max_val_latent (int): The range coder AC_MAX_VAL parameters for entropy coding the latents
    """

    n_bytes_header = 0
    n_bytes_header += 2  # Number of bytes header
    n_bytes_header += 1  # Display index of the frame

    n_bytes_header += (
        1  # Context size and Hidden layer dimension ARM, n. hidden layers ARM
    )

    n_bytes_header += 1 # (n_ups_kernel << 4)|(ups_k_size)
    n_bytes_header += 1 # (n_ups_preconcat_kernel << 4)|(ups_preconcat_k_size)

    n_bytes_header += 1  # Number of synthesis branches
    n_bytes_header += 1  # Number hidden layer Synthesis per branch
    # Hidden Synthesis layer out#, kernelsz, mode+nonlinearity
    n_bytes_header += 3 * len(frame_encoder.coolchic_encoder_param.layers_synthesis)
    n_bytes_header += 1  # Flow gain


    n_bytes_header += 2  # AC_MAX_VAL for neural networks
    n_bytes_header += 2  # AC_MAX_VAL for the latent variables
    n_bytes_header += 1     # hls_sig_blksize

    n_bytes_header += 1  # Index of the quantization step weight ARM
    n_bytes_header += 1  # Index of the quantization step bias ARM
    n_bytes_header += 1  # Index of the quantization step weight Upsampling
    n_bytes_header += 1  # Index of the quantization step bias Upsampling
    n_bytes_header += 1  # Index of the quantization step weight Synthesis
    n_bytes_header += 1  # Index of the quantization step bias Synthesis

    n_bytes_header += 1  # Index of scale entropy coding weight ARM
    n_bytes_header += 1  # Index of scale entropy coding bias ARM
    n_bytes_header += 1  # Index of scale entropy coding weight Upsampling
    n_bytes_header += 1  # Index of scale entropy coding bias Upsampling
    n_bytes_header += 1  # Index of scale entropy coding weight Synthesis
    n_bytes_header += 1  # Index of scale entropy coding bias Synthesis

    n_bytes_header += 2  # Number of bytes for weight ARM
    n_bytes_header += 2  # Number of bytes for bias ARM
    n_bytes_header += 2  # Number of bytes for weight Upsampling
    n_bytes_header += 2  # Index of scale entropy coding bias Upsampling
    n_bytes_header += 2  # Number of bytes for weight Synthesis
    n_bytes_header += 2  # Number of bytes for bias Synthesis

    n_bytes_header += 1  # Number of latent resolutions
    n_bytes_header += 1  # Number of 2D latent grids
    n_bytes_header += 1 * len(
        frame_encoder.coolchic_encoder.latent_grids
    )  # Number of feature maps for each latent resolutions
    n_bytes_header += 3 * len(
        n_bytes_per_latent
    )  # Number of bytes for each 2D latent grid


    byte_to_write = b""
    byte_to_write += n_bytes_header.to_bytes(2, byteorder="big", signed=False)
    byte_to_write += frame.display_order.to_bytes(1, byteorder="big", signed=False)


    # Since the dimension of the hidden layer and of the context is always a multiple of
    # 8, we can spare 3 bits by dividing it by 8
    assert frame_encoder.coolchic_encoder_param.dim_arm // 8 < 2**4, (
        f"Number of context pixels"
        f" and dimension of the hidden layer for the arm must be inferior to {2 ** 4 * 8}. Found"
        f" {frame_encoder.coolchic_encoder_param.dim_arm}"
    )

    assert frame_encoder.coolchic_encoder_param.n_hidden_layers_arm < 2**4, (
        f"Number of hidden layers"
        f" for the ARM should be inferior to {2 ** 4}. Found "
        f"{frame_encoder.coolchic_encoder_param.n_hidden_layers_arm}"
    )

    byte_to_write += (
        (frame_encoder.coolchic_encoder_param.dim_arm // 8) * 2**4
        + frame_encoder.coolchic_encoder_param.n_hidden_layers_arm
    ).to_bytes(1, byteorder="big", signed=False)

    byte_to_write += (
        # (frame_encoder.coolchic_encoder_param.n_ups_kernel<<4)|(frame_encoder.coolchic_encoder_param.ups_k_size)
        ((frame_encoder.coolchic_encoder_param.latent_n_grids-1)<<4)|(frame_encoder.coolchic_encoder_param.ups_k_size)
    ).to_bytes(1, byteorder="big", signed=False)
    byte_to_write += (
        # (frame_encoder.coolchic_encoder_param.n_ups_preconcat_kernel<<4)|(frame_encoder.coolchic_encoder_param.ups_preconcat_k_size)
        ((frame_encoder.coolchic_encoder_param.latent_n_grids-1)<<4)|(frame_encoder.coolchic_encoder_param.ups_preconcat_k_size)
    ).to_bytes(1, byteorder="big", signed=False)

    # Continue to send this byte for compatibility
    _dummy_n_synth_branch = 1
    byte_to_write += (
        _dummy_n_synth_branch  # frame_encoder.coolchic_encoder_param.n_synth_branch
    ).to_bytes(1, byteorder="big", signed=False)
    byte_to_write += len(
        frame_encoder.coolchic_encoder_param.layers_synthesis
    ).to_bytes(1, byteorder="big", signed=False)
    # If no hidden layers in the Synthesis, frame_encoder.coolchic_encoder_param.layers_synthesis is an empty list. So write 0
    for layer_spec in frame_encoder.coolchic_encoder_param.layers_synthesis:
        out_ft, k_size, mode, non_linearity = layer_spec.split("-")
        byte_to_write += int(out_ft).to_bytes(1, byteorder="big", signed=False)
        byte_to_write += int(k_size).to_bytes(1, byteorder="big", signed=False)
        byte_to_write += (
            _POSSIBLE_SYNTHESIS_MODE.index(mode) * 16
            + _POSSIBLE_SYNTHESIS_NON_LINEARITY.index(non_linearity)
        ).to_bytes(1, byteorder="big", signed=False)

    flow_gain = frame_encoder.inter_coding_module.flow_gain

    if isinstance(flow_gain, float):
        assert flow_gain.is_integer(), f"Flow gain should be integer, found {flow_gain}"
        flow_gain = int(flow_gain)

    assert (
        flow_gain >= 0 and flow_gain <= 255
    ), f"Flow gain should be in [0, 255], found {flow_gain}"
    byte_to_write += flow_gain.to_bytes(1, byteorder="big", signed=False)



    if ac_max_val_nn > MAX_AC_MAX_VAL:
        print("AC_MAX_VAL NN is too big!")
        print(f"Found {ac_max_val_nn}, should be smaller than {MAX_AC_MAX_VAL}")
        print("Exiting!")
        return
    if ac_max_val_latent > MAX_AC_MAX_VAL:
        print("AC_MAX_VAL latent is too big!")
        print(f"Found {ac_max_val_latent}, should be smaller than {MAX_AC_MAX_VAL}")
        print("Exiting!")
        return

    byte_to_write += ac_max_val_nn.to_bytes(2, byteorder="big", signed=False)
    byte_to_write += ac_max_val_latent.to_bytes(2, byteorder="big", signed=False)
    byte_to_write += hls_sig_blksize.to_bytes(1, byteorder='big', signed=True)

    for nn_name in ["arm", "upsampling", "synthesis"]:
        for nn_param in ["weight", "bias"]:
            cur_q_step_index = q_step_index_nn.get(nn_name).get(nn_param)
            byte_to_write += cur_q_step_index.to_bytes(
                1, byteorder="big", signed=False
            )

    for nn_name in ["arm", "upsampling", "synthesis"]:
        for nn_param in ["weight", "bias"]:
            cur_q_step_index = q_step_index_nn.get(nn_name).get(nn_param)
            cur_scale_index = scale_index_nn.get(nn_name).get(nn_param)
            byte_to_write += cur_scale_index.to_bytes(1, byteorder="big", signed=False)

    for nn_name in ["arm", "upsampling", "synthesis"]:
        for nn_param in ["weight", "bias"]:
            cur_q_step_index = q_step_index_nn.get(nn_name).get(nn_param)
            cur_n_bytes = n_bytes_nn.get(nn_name).get(nn_param)
            if cur_n_bytes > MAX_AC_MAX_VAL:
                print(f"Number of bytes for {nn_name} {nn_param} is too big!")
                print(f"Found {cur_n_bytes}, should be smaller than {MAX_AC_MAX_VAL}")
                print("Exiting!")
                return
            byte_to_write += cur_n_bytes.to_bytes(2, byteorder="big", signed=False)

    byte_to_write += frame_encoder.coolchic_encoder_param.latent_n_grids.to_bytes(
        1, byteorder="big", signed=False
    )
    byte_to_write += len(n_bytes_per_latent).to_bytes(1, byteorder="big", signed=False)

    for i, latent_i in enumerate(frame_encoder.coolchic_encoder.latent_grids):
        n_ft_i = latent_i.size()[1]
        if n_ft_i > 2**8 - 1:
            print(f"Number of feature maps for latent {i} is too big!")
            print(f"Found {n_ft_i}, should be smaller than {2 ** 8 - 1}")
            print("Exiting!")
            return
        byte_to_write += n_ft_i.to_bytes(1, byteorder="big", signed=False)

    for i, v in enumerate(n_bytes_per_latent):
        if v > 2**24 - 1:
            print(f"Number of bytes for latent {i} is too big!")
            print(f"Found {v}, should be smaller than {2 ** 24 - 1}")
            print("Exiting!")
            return
        # for tmp in n_bytes_per_latent:
        byte_to_write += v.to_bytes(3, byteorder="big", signed=False)

    with open(header_path, "wb") as fout:
        fout.write(byte_to_write)

    if n_bytes_header != os.path.getsize(header_path):
        print("Invalid number of bytes in header!")
        print("expected", n_bytes_header)
        print("got", os.path.getsize(header_path))
        exit(1)
