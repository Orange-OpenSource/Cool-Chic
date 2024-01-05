# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


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

    ? Everything here is detailed in src/encoding_management/coding_structure.py
    [GOP TYPE, FRAME_DATA_TYPE, BITDEPTH]           1 byte (Everything condensed in one byte)
    [Intra Period]                                  1 byte (From 0 to 255)
    [P-period]                                      1 byte (From 0 to 255)
    ? ========================= GOP HEADER ========================= ?


Header for the frame (one by frame):
------------------------------------

    ? ======================== FRAME HEADER ======================== ?
    [Number of bytes used for the header]           2 bytes
    [Display index of the frame]                    1 byte (From 0. to 255.) Not absolutely necessary!
    [Number of row & column of context]             1 byte
    [Number of hidden layer ARM]                    1 byte  (e.g. 2 if archi='24,24')
    [Hidden layer dimension ARM]                    1 byte  (e.g. 24 if archi='24,24')
    [Upsampling layer kernel size]                  1 byte  (From 0 to 255)
    [Number of layer Synthesis]                     1 byte  (e.g. 2 if archi='24,24')
    [Synthesis layer 0 description ]                3 bytes  (Contains out_ft / k_size / layer_type / non-linearity)
                        ...
    [Synthesis layer N - 1 description ]            3 bytes  (Contains out_ft / k_size / layer_type / non-linearity)


    [Index of quantization step weight ARM]         1 bytes (From 0 to 255)
    [Index of quantization step bias ARM]           1 bytes (From 0 to 255)
    [Index of quantization step weight Upsampling]  1 bytes (From 0 to 255)
    [Index of quantization step bias Upsampling]    1 bytes (From 0 to 255)
    [Index of quantization step weight Synthesis]   1 bytes (From 0 to 255)
    [Index of quantization step bias Synthesis]     1 bytes (From 0 to 255)

    [Index of scale entropy coding weight ARM]      2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding bias ARM]        2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding weight Upsampling] 2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding bias Upsampling]   2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding weight Synthesis]2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding bias Synthesis]  2 bytes (From 0 to 2 ** 16 - 1)

    [Number of bytes used for weight ARM]           2 bytes (less than 65535 bytes)
    [Number of bytes used for bias ARM]             2 bytes (less than 65535 bytes)
    [Number of bytes used for weight Upsampling]           2 bytes (less than 65535 bytes)
    [Number of bytes used for bias Upsampling]             2 bytes (less than 65535 bytes)
    [Number of bytes used for weight Synthesis]     2 bytes (less than 65535 bytes)
    [Number of bytes used for bias Synthesis]       2 bytes (less than 65535 bytes)

    ! Here, we split the 3D latent grids into several 2D grids. For instance,
    ! if we have three latent resolutions:
    !       [1, H, W] ; [3, H / 2, W / 2] and [2, H / 4, W / 4]
    ! We have 1 + 3 + 2 = 6 2D latent grids but 3 latent resolutions
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
from encoding_management.coding_structure import FRAME_DATA_TYPE, GOP_TYPE, POSSIBLE_BITDEPTH
from models.coolchic_components.synthesis import Synthesis
from models.frame_encoder import FrameEncoder
from models.video_encoder import VideoEncoder
from utils.misc import MAX_AC_MAX_VAL, DescriptorCoolChic



# Quick & dirty copy and paste from encoding_management.coding_structure
_GOP_TYPE = ['RA', 'LDP']
_FRAME_DATA_TYPE = ['rgb', 'yuv420', 'yuv444']
_POSSIBLE_BITDEPTH = [8, 10]
_POSSIBLE_SYNTHESIS_MODE = [k for k in Synthesis.possible_mode]
_POSSIBLE_SYNTHESIS_NON_LINEARITY = [k for k in Synthesis.possible_non_linearity]


class GopHeader(TypedDict):
    n_bytes_header: int                 # Number of bytes for the header
    img_size: Tuple[int, int]           # Format: (height, width)
    gop_type: GOP_TYPE                  # RA or LDP
    frame_data_type: FRAME_DATA_TYPE    # RGB, YUV 4:2:0, YUV 4:4:4
    bitdepth: POSSIBLE_BITDEPTH         # 8 or 10
    intra_period: int                   # See coding_structure.py
    p_period: int                       # See coding_structure.py

def write_gop_header(video_encoder: VideoEncoder, header_path: str):
    """Write a frame header to a a file located at <header_path>.
    The structure of the header is described above.

    Args:
        video_encoder (VideoEncoder): Model from which info located in the header will be retrieved.
        header_path (str): Path of the file where the header is written.
    """

    n_bytes_header = 0
    n_bytes_header += 2     # Number of bytes header
    n_bytes_header += 2     # Image height
    n_bytes_header += 2     # Image width
    n_bytes_header += 1     # GOP type, frame data type, bitdepth
    n_bytes_header += 1     # intra period
    n_bytes_header += 1     # p period

    byte_to_write = b''
    byte_to_write += n_bytes_header.to_bytes(2, byteorder='big', signed=False)

    byte_to_write += video_encoder.img_size[-2].to_bytes(2, byteorder='big', signed=False)
    byte_to_write += video_encoder.img_size[-1].to_bytes(2, byteorder='big', signed=False)

    byte_to_write += (
        # Last three bits are for the bitdepth
        _POSSIBLE_BITDEPTH.index(video_encoder.bitdepth) * 2 ** 5 \
        # The three intermediates bits are for the frame data type
        + _FRAME_DATA_TYPE.index(video_encoder.frame_data_type) * 2 ** 2 \
        # First two bits are for the frame data type
        + _GOP_TYPE.index(video_encoder.coding_structure.gop_type)
    ).to_bytes(1, byteorder='big', signed=False)

    byte_to_write += video_encoder.coding_structure.intra_period.to_bytes(1, byteorder='big', signed=False)
    byte_to_write += video_encoder.coding_structure.p_period.to_bytes(1, byteorder='big', signed=False)

    with open(header_path, 'wb') as fout:
        fout.write(byte_to_write)

    if n_bytes_header != os.path.getsize(header_path):
        print('Invalid number of bytes in header!')
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
    with open(bitstream_path, 'rb') as fin:
        bitstream = fin.read()


    ptr = 0
    n_bytes_header = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2

    img_height = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2
    img_width = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2

    raw_value = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    bitdepth = _POSSIBLE_BITDEPTH[raw_value // 2 ** 5]
    frame_data_type = _FRAME_DATA_TYPE[(raw_value % 2 ** 5) // 2 ** 2]
    gop_type = _GOP_TYPE[raw_value % 2 ** 2]

    intra_period = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    p_period = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1

    header_info: GopHeader = {
        'n_bytes_header': n_bytes_header,
        'img_size': (img_height, img_width),
        'gop_type': gop_type,
        'frame_data_type': frame_data_type,
        'bitdepth': bitdepth,
        'intra_period': intra_period,
        'p_period': p_period,
    }

    print('\nContent of the GOP header:')
    print('---------------------------')
    for k, v in header_info.items():
        print(f'{k:>20}: {v}')
    print('         ---------------------')

    return header_info

class FrameHeader(TypedDict):
    """Define the dictionary containing the header information as a type."""
    n_bytes_header: int                 # Number of bytes for the header
    display_index: int                  # Display index of the frame
    latent_n_resolutions: int           # Number of different latent resolutions
    latent_n_2d_grid: int               # Number of 2D latent grids
    n_ft_per_latent: List[int]          # Number of feature maps for each latent resolutions
    n_bytes_per_latent: List[int]       # Number of bytes for each 2D latent
    img_size: Tuple[int, int]           # Format: (height, width)
    n_ctx_rowcol: int                   # Number of row & column of context used by the ARM
    layers_arm: List[int]               # Dimension of each hidden layer in the ARM
    upsampling_kernel_size: int         # Upsampling model kernel size
    layers_synthesis: List[str]         # Dimension of each hidden layer in the Synthesis
    q_step_index_nn: DescriptorCoolChic # Index of the quantization step used for the weight & bias of the NNs
    scale_index_nn: DescriptorCoolChic  # Index of the scale used for entropy code weight & bias of the NNs
    n_bytes_nn: DescriptorCoolChic      # Number of bytes for weight and bias of the NNs
    ac_max_val_nn: int                  # The range coder AC_MAX_VAL parameters for entropy coding the NNs
    ac_max_val_latent: int              # The range coder AC_MAX_VAL parameters for entropy coding the latents
    image_type: str                     # Image type from the list possible_image_type defined above

def write_frame_header(
    frame_encoder: FrameEncoder,
    header_path: str,
    n_bytes_per_latent: List[int],
    q_step_index_nn: DescriptorCoolChic,
    scale_index_nn: DescriptorCoolChic,
    n_bytes_nn: DescriptorCoolChic,
    ac_max_val_nn: int,
    ac_max_val_latent: int,
):
    """Write a frame header to a a file located at <header_path>.
    The structure of the header is described above.

    Args:
        model (frame_encoder): Model from which info located in the header will be retrieved.
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
    n_bytes_header += 2     # Number of bytes header
    n_bytes_header += 1     # Display index of the frame

    n_bytes_header += 1     # Number of row and column of context
    n_bytes_header += 1     # Number hidden layer ARM
    n_bytes_header += 1     # Hidden layer dimension ARM
    n_bytes_header += 1     # Upsampling kernel size
    n_bytes_header += 1     # Number hidden layer Synthesis
    # Hidden Synthesis layer out#, kernelsz, mode+nonlinearity
    n_bytes_header += 3 * len(frame_encoder.coolchic_encoder_param.layers_synthesis)

    n_bytes_header += 2     # AC_MAX_VAL for neural networks
    n_bytes_header += 2     # AC_MAX_VAL for the latent variables

    n_bytes_header += 1     # Index of the quantization step weight ARM
    n_bytes_header += 1     # Index of the quantization step bias ARM
    n_bytes_header += 1     # !! -1 => no bias # Index of the quantization step weight Upsampling
    n_bytes_header += 1     # Index of the quantization step bias Upsampling
    n_bytes_header += 1     # Index of the quantization step weight Synthesis
    n_bytes_header += 1     # Index of the quantization step bias Synthesis

    n_bytes_header += 2     # Index of scale entropy coding weight ARM
    n_bytes_header += 2     # Index of scale entropy coding bias ARM
    n_bytes_header += 2     # Index of scale entropy coding weight Upsampling
    n_bytes_header += 2     # Index of scale entropy coding weight Synthesis
    n_bytes_header += 2     # Index of scale entropy coding bias Synthesis

    n_bytes_header += 2     # Number of bytes for weight ARM
    n_bytes_header += 2     # Number of bytes for bias ARM
    n_bytes_header += 2     # Number of bytes for weight Upsampling
    n_bytes_header += 2     # Number of bytes for weight Synthesis
    n_bytes_header += 2     # Number of bytes for bias Synthesis

    n_bytes_header += 1     # Number of latent resolutions
    n_bytes_header += 1     # Number of 2D latent grids
    n_bytes_header += 1 * len(frame_encoder.coolchic_encoder.latent_grids)   # Number of feature maps for each latent resolutions
    n_bytes_header += 3 * len(n_bytes_per_latent)   # Number of bytes for each 2D latent grid

    byte_to_write = b''
    byte_to_write += n_bytes_header.to_bytes(2, byteorder='big', signed=False)
    byte_to_write += frame_encoder.frame.display_order.to_bytes(1, byteorder='big', signed=False)

    byte_to_write += frame_encoder.coolchic_encoder_param.n_ctx_rowcol.to_bytes(1, byteorder='big', signed=False)
    byte_to_write += len(frame_encoder.coolchic_encoder_param.layers_arm).to_bytes(1, byteorder='big', signed=False)
    # If no hidden layers in the ARM, model.param.layers_arm is an empty list. So write 0
    if len(frame_encoder.coolchic_encoder_param.layers_arm) == 0:
        byte_to_write += int(0).to_bytes(1, byteorder='big', signed=False)
    else:
        byte_to_write += frame_encoder.coolchic_encoder_param.layers_arm[0].to_bytes(1, byteorder='big', signed=False)

    byte_to_write += frame_encoder.coolchic_encoder_param.upsampling_kernel_size.to_bytes(1, byteorder='big', signed=False)

    byte_to_write += len(frame_encoder.coolchic_encoder_param.layers_synthesis).to_bytes(1, byteorder='big', signed=False)
    # If no hidden layers in the Synthesis, frame_encoder.coolchic_encoder_param.layers_synthesis is an empty list. So write 0
    for layer_spec in frame_encoder.coolchic_encoder_param.layers_synthesis:
        out_ft, k_size, mode, non_linearity = layer_spec.split('-')
        byte_to_write += int(out_ft).to_bytes(1, byteorder='big', signed=False)
        byte_to_write += int(k_size).to_bytes(1, byteorder='big', signed=False)
        byte_to_write += (_POSSIBLE_SYNTHESIS_MODE.index(mode)*16+_POSSIBLE_SYNTHESIS_NON_LINEARITY.index(non_linearity)).to_bytes(1, byteorder='big', signed=False)

    if ac_max_val_nn > MAX_AC_MAX_VAL:
        print(f'AC_MAX_VAL NN is too big!')
        print(f'Found {ac_max_val_nn}, should be smaller than {MAX_AC_MAX_VAL}')
        print(f'Exiting!')
        return
    if ac_max_val_latent > MAX_AC_MAX_VAL:
        print(f'AC_MAX_VAL latent is too big!')
        print(f'Found {ac_max_val_latent}, should be smaller than {MAX_AC_MAX_VAL}')
        print(f'Exiting!')
        return

    byte_to_write += ac_max_val_nn.to_bytes(2, byteorder='big', signed=False)
    byte_to_write += ac_max_val_latent.to_bytes(2, byteorder='big', signed=False)

    for nn_name in ['arm', 'upsampling', 'synthesis']:
        for nn_param in ['weight', 'bias']:
            cur_q_step_index = q_step_index_nn.get(nn_name).get(nn_param)
            if cur_q_step_index < 0:
                # hack -- ensure -1 translated to 255.
                byte_to_write += int(255).to_bytes(1, byteorder='big', signed=False)
            else:
                byte_to_write += cur_q_step_index.to_bytes(1, byteorder='big', signed=False)

    for nn_name in ['arm', 'upsampling', 'synthesis']:
        for nn_param in ['weight', 'bias']:
            cur_q_step_index = q_step_index_nn.get(nn_name).get(nn_param)
            if cur_q_step_index < 0:
                # no bias.
                continue
            cur_scale_index = scale_index_nn.get(nn_name).get(nn_param)
            byte_to_write += cur_scale_index.to_bytes(2, byteorder='big', signed=False)

    for nn_name in ['arm', 'upsampling', 'synthesis']:
        for nn_param in ['weight', 'bias']:
            cur_q_step_index = q_step_index_nn.get(nn_name).get(nn_param)
            if cur_q_step_index < 0:
                # no bias
                continue
            cur_n_bytes = n_bytes_nn.get(nn_name).get(nn_param)
            if cur_n_bytes > MAX_AC_MAX_VAL:
                print(f'Number of bytes for {nn_name} {nn_param} is too big!')
                print(f'Found {cur_n_bytes}, should be smaller than {MAX_AC_MAX_VAL}')
                print(f'Exiting!')
                return
            byte_to_write += cur_n_bytes.to_bytes(2, byteorder='big', signed=False)

    byte_to_write += frame_encoder.coolchic_encoder_param.latent_n_grids.to_bytes(1, byteorder='big', signed=False)
    byte_to_write += len(n_bytes_per_latent).to_bytes(1, byteorder='big', signed=False)

    for i, latent_i in enumerate(frame_encoder.coolchic_encoder.latent_grids):
        n_ft_i = latent_i.size()[1]
        if n_ft_i > 2 ** 8 - 1:
            print(f'Number of feature maps for latent {i} is too big!')
            print(f'Found {n_ft_i}, should be smaller than {2 ** 8 - 1}')
            print(f'Exiting!')
            return
        byte_to_write += n_ft_i.to_bytes(1, byteorder='big', signed=False)

    for i, v in enumerate(n_bytes_per_latent):
        if v > 2 ** 24 - 1:
            print(f'Number of bytes for latent {i} is too big!')
            print(f'Found {v}, should be smaller than {2 ** 24 - 1}')
            print(f'Exiting!')
            return
    # for tmp in n_bytes_per_latent:
        byte_to_write += v.to_bytes(3, byteorder='big', signed=False)

    with open(header_path, 'wb') as fout:
        fout.write(byte_to_write)

    if n_bytes_header != os.path.getsize(header_path):
        print('Invalid number of bytes in header!')
        print("expected", n_bytes_header)
        print("got", os.path.getsize(header_path))
        exit(1)

def read_frame_header(bitstream: bytes) -> FrameHeader:
    """Read the first few bytes of a bitstream file located at
    <bitstream_path> and parse the different information.

    Args:
        bitstream_path (str): Path where the bitstream is located.

    Returns:
        FrameHeader: The parsed info from the bitstream.
    """

    ptr = 0
    n_bytes_header = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2

    # img_height = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    # ptr += 2
    # img_width = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    # ptr += 2

    # img_type = possible_image_type[int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)]
    # ptr += 1

    display_index = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1

    n_ctx_rowcol = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1

    n_hidden_dim_arm = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    hidden_size_arm = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1

    upsampling_kernel_size = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1

    n_hidden_dim_synthesis = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    layers_synthesis = []
    for i in range(n_hidden_dim_synthesis):
        out_ft = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        ptr += 1
        kernel_size = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        ptr += 1
        mode_non_linearity = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        ptr += 1
        mode = _POSSIBLE_SYNTHESIS_MODE[mode_non_linearity // 16]
        non_linearity = _POSSIBLE_SYNTHESIS_NON_LINEARITY[mode_non_linearity % 16]
        layers_synthesis.append(f'{out_ft}-{kernel_size}-{mode}-{non_linearity}')

    ac_max_val_nn = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2
    ac_max_val_latent = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2

    q_step_index_nn: DescriptorCoolChic = {}
    for nn_name in ['arm', 'upsampling', 'synthesis']:
        q_step_index_nn[nn_name] = {}
        for nn_param in ['weight', 'bias']:
            q_step_index_nn[nn_name][nn_param] = int.from_bytes(
                bitstream[ptr: ptr + 1], byteorder='big', signed=False
            )
            # Hack -- 255 -> -1 for upsampling bias.  Indiciating no bias.
            if q_step_index_nn[nn_name][nn_param] == 255:
                q_step_index_nn[nn_name][nn_param] = -1
                print("got -1:", nn_name, nn_param)
            ptr += 1

    scale_index_nn: DescriptorCoolChic = {}
    for nn_name in ['arm', 'upsampling', 'synthesis']:
        scale_index_nn[nn_name] = {}
        for nn_param in ['weight', 'bias']:
            if q_step_index_nn[nn_name][nn_param] < 0:
                scale_index_nn[nn_name][nn_param] = -1
            else:
                scale_index_nn[nn_name][nn_param] = int.from_bytes(
                    bitstream[ptr: ptr + 2], byteorder='big', signed=False
                )
                ptr += 2

    n_bytes_nn: DescriptorCoolChic = {}
    for nn_name in ['arm', 'upsampling', 'synthesis']:
        n_bytes_nn[nn_name] = {}
        for nn_param in ['weight', 'bias']:
            if q_step_index_nn[nn_name][nn_param] < 0:
                n_bytes_nn[nn_name][nn_param] = -1
            else:
                n_bytes_nn[nn_name][nn_param] = int.from_bytes(
                    bitstream[ptr: ptr + 2], byteorder='big', signed=False
                )
                ptr += 2

    latent_n_resolutions = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    latent_n_2d_grid = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1

    n_ft_per_latent = []
    for _ in range(latent_n_resolutions):
        n_ft_per_latent.append(
            int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        )
        ptr += 1


    n_bytes_per_latent = []
    for _ in range(latent_n_2d_grid):
        n_bytes_per_latent.append(
            int.from_bytes(bitstream[ptr: ptr + 3], byteorder='big', signed=False)
        )
        ptr += 3

    header_info: FrameHeader = {
        'n_bytes_header': n_bytes_header,
        'latent_n_resolutions': latent_n_resolutions,
        'latent_n_2d_grid': latent_n_2d_grid,
        'n_bytes_per_latent': n_bytes_per_latent,
        'n_ft_per_latent': n_ft_per_latent,
        'n_ctx_rowcol': n_ctx_rowcol,
        'layers_arm': [hidden_size_arm for _ in range(n_hidden_dim_arm)],
        'upsampling_kernel_size': upsampling_kernel_size,
        'layers_synthesis': layers_synthesis,
        'q_step_index_nn': q_step_index_nn,
        'scale_index_nn': scale_index_nn,
        'n_bytes_nn': n_bytes_nn,
        'ac_max_val_nn': ac_max_val_nn,
        'ac_max_val_latent': ac_max_val_latent,
        'display_index': display_index
    }

    print('\nContent of the frame header:')
    print('------------------------------')
    for k, v in header_info.items():
        print(f'{k:>20}: {v}')
    print('         ------------------------')

    return header_info
