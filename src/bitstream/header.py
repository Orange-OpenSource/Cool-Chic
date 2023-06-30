# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
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
    MLPs hidden layers must be of constant size e.g. 24,24 or 16,16

1. Header

    [Number of bytes used for the header]           2 bytes
    [Image height]                                  2 bytes
    [Image width]                                   2 bytes
    [Image type]                                    1 byte

    [Number of row & column of context]             1 byte
    [Number of hidden layer ARM]                    1 byte  (e.g. 2 if archi='24,24')
    [Hidden layer dimension ARM]                    1 byte  (e.g. 24 if archi='24,24')
    [Number of hidden layer Synthesis]              1 byte  (e.g. 2 if archi='24,24')
    [Hidden layer dimension Synthesis]              1 byte  (e.g. 24 if archi='24,24')

    [Index of quantization step weight ARM]         1 bytes (From 0 to 255)
    [Index of quantization step bias ARM]           1 bytes (From 0 to 255)
    [Index of quantization step weight Synthesis]   1 bytes (From 0 to 255)
    [Index of quantization step bias Synthesis]     1 bytes (From 0 to 255)

    [Index of scale entropy coding weight ARM]      2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding bias ARM]        2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding weight Synthesis]2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding bias Synthesis]  2 bytes (From 0 to 2 ** 16 - 1)

    [Number of bytes used for weight ARM]           2 bytes (less than 65535 bytes)
    [Number of bytes used for bias ARM]             2 bytes (less than 65535 bytes)
    [Number of bytes used for weight Synthesis]     2 bytes (less than 65535 bytes)
    [Number of bytes used for bias Synthesis]       2 bytes (less than 65535 bytes)

    [Number of latent grids]                        1 byte
    [Number of bytes used for latent grid 0]        3 bytes (less than 16777215 bytes)
    [Number of bytes used for latent grid 1]        3 bytes (less than 16777215 bytes)
                        ...
    [Number of bytes used for latent grid N - 1]    3 bytes (less than 16777215 bytes)
"""


import os
from typing import Dict, List, Tuple, TypedDict
from models.cool_chic import CoolChicEncoder
from utils.constants import MAX_AC_MAX_VAL
from utils.data_structure import DescriptorCoolChic

synthesis_possible_modes           = ['linear', 'residual', 'attention' ]
synthesis_possible_non_linearities = ['none', 'relu', 'leakyrelu' ]
possible_image_type                = ['rgb444', 'yuv420_8b', 'yuv420_10b']

class HeaderInfo(TypedDict):
    """Define the dictionary containing the header information as a type."""
    n_bytes_header: int                 # Number of bytes for the header
    latent_n_grids: int                 # Number of different latent grid
    n_bytes_per_latent: List[int]       # Number of bytes for each latent grid
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

def write_header(
    model: CoolChicEncoder,
    header_path: str,
    n_bytes_per_latent: List[int],
    q_step_index_nn: DescriptorCoolChic,
    scale_index_nn: DescriptorCoolChic,
    n_bytes_nn: DescriptorCoolChic,
    ac_max_val_nn: int,
    ac_max_val_latent: int,
):
    """Write a header to a a file located at <header_path>.
    The structure of the header is described above.

    Args:
        model (CoolChicEncoder): Model from which info located in the header
            will be retrieved.
        header_path (str): Path of the file where the header is written.
        n_bytes_per_latent (List[int]): Indicates the number of bytes for each
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
    n_bytes_header += 2     # Image height
    n_bytes_header += 2     # Image width
    n_bytes_header += 1     # Image type index
    n_bytes_header += 1     # Number of row and column of context

    n_bytes_header += 1     # Number hidden layer ARM
    n_bytes_header += 1     # Hidden layer dimension ARM
    n_bytes_header += 1     # Upsampling kernel size
    n_bytes_header += 1     # Number hidden layer Synthesis
    n_bytes_header += 3 * len(model.param.layers_synthesis) # Hidden Synthesis layer out#, kernelsz, mode+nonlinearity

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

    n_bytes_header += 1     # Number of latent grids
    n_bytes_header += 3 * len(n_bytes_per_latent)   # Number of bytes for each latent grid

    byte_to_write = b''
    byte_to_write += n_bytes_header.to_bytes(2, byteorder='big', signed=False)
    byte_to_write += model.param.img_size[0].to_bytes(2, byteorder='big', signed=False)
    byte_to_write += model.param.img_size[1].to_bytes(2, byteorder='big', signed=False)
    byte_to_write += possible_image_type.index(model.param.img_type).to_bytes(1, byteorder='big', signed=False)
    byte_to_write += model.param.n_ctx_rowcol.to_bytes(1, byteorder='big', signed=False)

    byte_to_write += len(model.param.layers_arm).to_bytes(1, byteorder='big', signed=False)
    # If no hidden layers in the ARM, model.param.layers_arm is an empty list. So write 0
    if len(model.param.layers_arm) == 0:
        byte_to_write += int(0).to_bytes(1, byteorder='big', signed=False)
    else:
        byte_to_write += model.param.layers_arm[0].to_bytes(1, byteorder='big', signed=False)

    byte_to_write += model.param.upsampling_kernel_size.to_bytes(1, byteorder='big', signed=False)

    byte_to_write += len(model.param.layers_synthesis).to_bytes(1, byteorder='big', signed=False)
    # If no hidden layers in the Synthesis, model.param.layers_synthesis is an empty list. So write 0
    for layer_spec in model.param.layers_synthesis:
        out_ft, k_size, mode, non_linearity = layer_spec.split('-')
        byte_to_write += int(out_ft).to_bytes(1, byteorder='big', signed=False)
        byte_to_write += int(k_size).to_bytes(1, byteorder='big', signed=False)
        byte_to_write += (synthesis_possible_modes.index(mode)*16+synthesis_possible_non_linearities.index(non_linearity)).to_bytes(1, byteorder='big', signed=False)

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

    byte_to_write += model.param.latent_n_grids.to_bytes(1, byteorder='big', signed=False)

    for i, v in enumerate(n_bytes_per_latent):
        if v > 2 ** 24 - 1:
            print(f'Number of bytes for latent {i} is too big!')
            print(f'Found {v}, should be smaller than {2 ** 24 - 1}')
            print(f'Exiting!')
            return

    for tmp in n_bytes_per_latent:
        byte_to_write += tmp.to_bytes(3, byteorder='big', signed=False)

    with open(header_path, 'wb') as fout:
        fout.write(byte_to_write)

    if n_bytes_header != os.path.getsize(header_path):
        print('Invalid number of bytes in header!')
        print("expected", n_bytes_header)
        print("got", os.path.getsize(header_path))
        exit(1)

def read_header(bitstream_path: str) -> HeaderInfo:
    """Read the first few bytes of a bitstream file located at
    <bitstream_path> and parse the different information.

    Args:
        bitstream_path (str): Path where the bitstream is located.

    Returns:
        HeaderInfo: The parsed info from the bitstream.
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

    img_type = possible_image_type[int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)]
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
        mode = synthesis_possible_modes[mode_non_linearity // 16]
        non_linearity = synthesis_possible_non_linearities[mode_non_linearity % 16]
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
                # print("got -1:", nn_name, nn_param)
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

    latent_n_grids = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1

    n_bytes_per_latent = []
    for _ in range(latent_n_grids):
        n_bytes_per_latent.append(
            int.from_bytes(bitstream[ptr: ptr + 3], byteorder='big', signed=False)
        )
        ptr += 3

    header_info: HeaderInfo = {
        'n_bytes_header': n_bytes_header,
        'latent_n_grids': latent_n_grids,
        'n_bytes_per_latent': n_bytes_per_latent,
        'img_size': (img_height, img_width),
        'n_ctx_rowcol': n_ctx_rowcol,
        'layers_arm': [hidden_size_arm for _ in range(n_hidden_dim_arm)],
        'upsampling_kernel_size': upsampling_kernel_size,
        'layers_synthesis': layers_synthesis,
        'q_step_index_nn': q_step_index_nn,
        'scale_index_nn': scale_index_nn,
        'n_bytes_nn': n_bytes_nn,
        'ac_max_val_nn': ac_max_val_nn,
        'ac_max_val_latent': ac_max_val_latent,
        'img_type': img_type,
    }

    print('\nContent of the header:')
    print('----------------------')
    for k, v in header_info.items():
        print(f'{k:>20}: {v}')
    print('         ----------------')

    return header_info
