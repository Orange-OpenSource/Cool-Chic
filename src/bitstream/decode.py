# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import glob
import subprocess
from typing import Tuple
import numpy as np
import torch
import time
import math
import torch.nn.functional as F

from torch import nn, Tensor
from bitstream.header import read_frame_header, read_gop_header
from bitstream.range_coder import RangeCoder
from encoding_management.coding_structure import CodingStructure, FrameData
from models.inter_coding_module import InterCodingModule
from utils.yuv import convert_420_to_444, convert_444_to_420, yuv_dict_clamp
from models.coolchic_components.arm import Arm, get_mu_scale
from utils.misc import DescriptorNN, POSSIBLE_Q_STEP_ARM_NN, POSSIBLE_Q_STEP_SYN_NN, POSSIBLE_SCALE_NN, FIXED_POINT_FRACTIONAL_MULT, ARMINT
from models.coolchic_components.upsampling import Upsampling
from models.coolchic_components.synthesis import Synthesis
from visu.utils import save_visualisation


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


def compute_offset(x: Tensor, mask_size: int) -> Tensor:
    """Compute the offset vector which will be applied to the 1d indices of
    a feature in order to extract its context.
    Let us suppose that we have n_ctx_rowcol = 1 and the following feature
    maps:

                                    0 0 0 0 0 0 0
        a b c d e   => padding =>   0 a b c d e 0
        f g h i j                   0 f g h i j 0
                                    0 0 0 0 0 0 0

        Decoding of pixel g uses the context {a, b, c, f}. If we expresses that as
    1d indices, we have:

    0   1   2   3   4   5   6
    7   8   9   10  11  12  13
    14  15  16  17  18  19  20
    21  22  23  24  25  26  27

    That is, coding of pixel g (index 16) uses the context pixels of indices {8, 9, 15}.
    The offset vector for this case is the one such that: 16 - offset = {8, 9, 15}:
        offset = {8, 7, 1}

    Args:
        x (Tensor): The feature maps to decode. Used only for its size.
        mask_size (int): Size of the mask (mask_size = 2 * n_ctx_rowcol + 1).

    Returns:
        Tensor: The offset vector
    """
    # Padding required by the number of context pixels
    pad = int((mask_size - 1) / 2)
    # Number of input for the ARM MLP
    n_context_pixel = int((mask_size ** 2 - 1) / 2)

    # Size of the actual and padded feature maps.
    H, W = x.size()[-2:]
    W_pad = W + 2 * pad


    # The four lines below are the equivalent of the following for loops:
    # For n_ctx_row_col = 3 we generate the following tensor:
    # idx_row = 3 - [0 0 0 0 0 0 0 1 1 1 1 1 1 1 2 2 2 2 2 2 2 3 3 3]
    # idx_col = 3 - [0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2]
    # Then, offset = idx_col + idx_row * W_pad indicates the offset, relative
    # to a current index, of the set of neighbor indices.
    idx_row = pad - torch.arange(0, pad + 1, device=x.device).repeat_interleave(mask_size)
    idx_col = pad - torch.arange(0, mask_size, device=x.device).repeat(pad + 1)
    idx_row = idx_row[:n_context_pixel]
    idx_col = idx_col[:n_context_pixel]

    # # From 0 to 3 included in case of mask_size = 7 (i.e. n_ctx_rowcol = 3)
    # for idx_row in range(pad + 1):
    #     # Last line of the context, only half of the mask
    #     if idx_row == pad:
    #         list_indices_col = [i for i in range(pad)]
    #     else:
    #         list_indices_col = [i for i in range(mask_size)]

    #     for idx_col in list_indices_col:
    #         offset.append((pad - idx_col) + (pad - idx_row) * W_pad)

    offset = idx_col + idx_row * W_pad
    return offset


def unpad_to_pad_index_1d(idx: Tensor, pad: int, W: int) -> Tensor:
    """Convert a tensor of 1d indices for a flatten 2D feature map to
    the 1d indices for the same feature maps with padding.

    Let us suppose that idx are the indices for the following feature maps,
    with n_ctx_row_col = pad = 1:
    Values:
    =======
                                    0 0 0 0 0 0 0
        a b c d e   => padding =>   0 a b c d e 0
        f g h i j                   0 f g h i j 0
                                    0 0 0 0 0 0 0

    Indices:
    ========
                                    0   1   2   3   4   5   6
        0 1 2 3 4   => padding =>   7   8   9   10  11  12  13
        5 6 7 8 9                   14  15  16  17  18  19  20
                                    21  22  23  24  25  26  27

    Pixel b has index 1 in the non-padded version, but the index 9
    in the padded version.

    Args:
        idx (Tensor): 1d tensor of the non-padded indices
        pad (int): Padding on each side (equals to n_ctx_rowcol)
        W (int): Width of the **non-padded** 2d feature map, whose flattened
            version is indexed by idx.

    Returns:
        Tensor: 1d tensor of the padded indices
    """
    # We add pad columns of zero pixel on both side of the feature map
    W_pad = W + 2 * pad

    idx_w_pad = idx + pad + pad * W_pad + (idx // W) * 2 * pad
    #                  ^        ^                    ^
    #                  |        |                    |_________ Each new rows (idx // W) brings pad columns on both sides
    #                  |        |______________________________ The first rows of padding
    #                  |_______________________________________ The first columns of padding
    return idx_w_pad


def pad_to_unpad_index_1d(idx_w_pad: Tensor, pad: int, W_pad: int) -> Tensor:
    """Inverse function of unpad_to_pad_index_1d:
        pad_to_unpad_index_1d(unpad_to_pad_index_1d(x)) = x

    Convert a tensor of 1d indices for a **padded** flatten 2D feature map to
    the 1d indices for the same feature maps **without** padding.

    Let us suppose that idx are the indices for the following feature maps,
    with n_ctx_row_col = pad = 1:
    Values:
    =======
                                    0 0 0 0 0 0 0
        a b c d e   => padding =>   0 a b c d e 0
        f g h i j                   0 f g h i j 0
                                    0 0 0 0 0 0 0

    Indices:
    ========
                                    0   1   2   3   4   5   6
        0 1 2 3 4   => padding =>   7   8   9   10  11  12  13
        5 6 7 8 9                   14  15  16  17  18  19  20
                                    21  22  23  24  25  26  27

    Pixel b has index 1 in the non-padded version, but the index 9
    in the padded version.

    Args:
        idx_w_pad (Tensor): 1d tensor of the padded indices
        pad (int): Padding on each side (equals to n_ctx_rowcol)
        W_pad (int): Width of the **padded** 2d feature map, whose flattened
            version is indexed by idx_w_pad.

    Returns:
        Tensor: 1d tensor of the **non-padded** indices
    """
    # # We remove pad columns of zero pixel on both side of the padded feature map
    # W = W_pad - 2 * pad
    idx = idx_w_pad - pad - pad * W_pad - ((idx_w_pad - pad * W_pad) // W_pad) * 2 * pad
    #                  ^        ^                    ^
    #                  |        |                    |_________ Each additional new rows after the first padded ones
    #                  |        |                               ((idx_w_pad - pad * W_pad) // W_pad) brings pad columns on both sides
    #                  |        |
    #                  |        |______________________________ The first rows of padding
    #                  |_______________________________________ The first columns of padding
    return idx

def fast_get_neighbor(x: Tensor, mask_size: int, offset: Tensor, idx: Tensor, w_grid: int) -> Tensor:
    """Use the unfold function to extract the neighbors of each pixel in x.

    Args:
        x (Tensor): [c_grid * h_grid * w_grid] flattened feature map from which we wish to extract the
            neighbors
        mask_size (int): Virtual size of the kernel around the current coded latent.
            mask_size = 2 * n_ctx_rowcol - 1
        offset (Tensor): [N_context_pixels] 1D tensor containing the indices of the N
            context pixels relatively to a given decoded pixels. See description
            in the function compute_offset above.
        idx (Tensor): 1d indices, indexing the 1d flattened feature map x. We want to
            extract the neighbors of the pixels located at x[idx]
        w_grid (int): width of the non-padded feature maps

    Returns:
        torch.tensor: [H * W, floor(N ** 2 / 2) - 1] the spatial neighbors
            the floor(N ** 2 / 2) - 1 neighbors of each H * W pixels.
    """
    pad = int((mask_size - 1) / 2)
    n_context_pixel = int((mask_size ** 2 - 1) / 2)

    # Convert indices in the image to indices with padding
    # Then repeat it to be of the same size than offset size (i.e. N spatial context)
    # [idx.numel()] => [idx.numel(), 1] => [idx.numel(), n_context_pixel]
    idx_w_pad = unpad_to_pad_index_1d(idx, pad, w_grid).view(-1, 1).repeat(1, n_context_pixel)
    neighbor_idx = (idx_w_pad - offset).flatten()

    # Select the proper indices into x_pad
    neighbor = torch.index_select(x, dim=0, index=neighbor_idx).view(-1, n_context_pixel)
    return neighbor


def decode_network(
    empty_module: nn.Module,
    bitstream_path: DescriptorNN,
    q_step_nn: DescriptorNN,
    scale_nn: DescriptorNN,
    ac_max_val: int,
) -> nn.Module:
    """Decode a neural network from a bitstream. The idea is to iterate
    on all the parameters of <empty_module>, filling it with values read
    from the bitstream.

    Args:
        empty_module (nn.Module): An empty (i.e. randomly initialized) instance
            of the network to load.
        bitstream_path (str): Weight and bias will be found at
            <bitstream_path>_weight and <bitstream_path>_arm
        q_step_nn (DescriptorNN): Describe the quantization steps used
            for the weight and bias of the network.
        scale_nn (DescriptorNN): Describe the scale parameters used
            for entropy coding of the weight and bias of the network.
        ac_max_val (int): Data are in [-ac_max_val, ac_max_val - 1]

    Returns:
        nn.Module: The decoded module
    """
    have_bias = q_step_nn.bias > 0

    # Instantiate two range coder objects to decode simultaneously weight and bias
    range_coder_nn_weight = RangeCoder(
        0,      # 0 because we don't have a ctx_row_col
        AC_MAX_VAL = ac_max_val
    )
    range_coder_nn_weight.load_bitstream(bitstream_path.weight)

    if have_bias:
        range_coder_nn_bias = RangeCoder(
            0,      # 0 because we don't have a ctx_row_col
            AC_MAX_VAL = ac_max_val
        )
        range_coder_nn_bias.load_bitstream(bitstream_path.bias)

    loaded_param = {}
    for k, v in empty_module.named_parameters():
        if k.endswith('.w') or k.endswith('.weight'):
            cur_scale = scale_nn.weight
            cur_q_step = q_step_nn.weight
            cur_param = range_coder_nn_weight.decode(
                torch.zeros_like(v.flatten()),
                torch.ones_like(v.flatten()) * cur_scale
            )
        elif k.endswith('.b') or k.endswith('.bias'):
            cur_scale = scale_nn.bias
            cur_q_step = q_step_nn.bias
            cur_param = range_coder_nn_bias.decode(
                torch.zeros_like(v.flatten()),
                torch.ones_like(v.flatten()) * cur_scale
            )
        else:
            # Ignore network parameters whose name does not end with '.w', '.b', '.weight', '.bias'
            continue

        # Don't forget inverse quantization!
        loaded_param[k] = cur_param.reshape_as(v)  * cur_q_step

    empty_module.load_state_dict(loaded_param)
    return empty_module

@torch.no_grad()
def decode_frame(bitstream_path: str, bitstream: bytes, img_size: Tuple[int, int]) -> Tuple[Tensor, bytes]:
    """Decode a bitstream located at <bitstream_path> and save the decoded image
    at <output_path>

    Args:
        bitstream_path (str): Absolute path of the bitstream. We keep that to output some temporary file
            at the same location than the bitstream itself
        bitstream (bytes): The bytes composing the bitstream
        img_size (Tuple[int, int]): Height and width of the frame

    Returns:
        Tensor: The decoded image with shape [1, 3, H, W] in [0., 1.]. Normalization, and conversion
            into 4:2:0 does **not** happen in this function
        bytes: The remaining bytes of the bitstream (i.e. for the following frames).
    """
    start_time = time.time()
    torch.use_deterministic_algorithms(True)

    # ========================== Parse the header =========================== #
    header_info = read_frame_header(bitstream)
    # ========================== Parse the header =========================== #

    # =================== Split the intermediate bitstream ================== #
    # The constriction range coder only works when we decode each latent grid
    # from a dedicated bitstream file.
    # The idea here is to split the single bitstream file to N files
    # bitstream_i, one for each latent grid

    # Read all the bits and discard the header
    bitstream = bitstream[header_info.get('n_bytes_header'): ]

    # For each module and each parameter type, keep the first bytes, write them
    # in a standalone file and keep the remainder of the bitstream in bitstream
    # Next parameter of next module will start at the first element of the bitstream.
    for cur_module_name in ['arm', 'upsampling', 'synthesis']:
        for parameter_type in ['weight', 'bias']:
            cur_n_bytes = header_info.get('n_bytes_nn')[cur_module_name][parameter_type]

            # Write the first byte in a dedicated file
            if cur_n_bytes > 0:
                bytes_for_current_parameters = bitstream[:cur_n_bytes]
                current_bitstream_file = f'{bitstream_path}_{cur_module_name}_{parameter_type}'
                with open(current_bitstream_file, 'wb') as f_out:
                    f_out.write(bytes_for_current_parameters)

                # Keep the rest of the bitstream for next loop
                bitstream = bitstream[cur_n_bytes: ]

    # For each latent grid: keep the first bytes to decode the current latent
    # and store the rest in bitstream. Latent i + 1 will start at the first
    # element of bitstream
    ctr_2d_ft = 0
    for index_lat_resolution in range(header_info.get('latent_n_resolutions')):

        # No 2d feature for this latent resolution... we still write an empty binary file
        # and increment the counter
        if header_info.get('n_ft_per_latent')[index_lat_resolution] == 0:
            # Write the first bytes in a dedicated file
            cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
            subprocess.call(f'touch {cur_latent_bitstream}', shell=True)
            ctr_2d_ft += 1
            continue


        for index_lat_feature in range(header_info.get('n_ft_per_latent')[index_lat_resolution]):
            cur_n_bytes = header_info.get('n_bytes_per_latent')[ctr_2d_ft]

            # Write the first bytes in a dedicated file
            cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
            bytes_for_current_latent = bitstream[: cur_n_bytes]
            with open(cur_latent_bitstream, 'wb') as f_out:
                f_out.write(bytes_for_current_latent)

            # Keep the rest of the bitstream for next loop
            bitstream = bitstream[cur_n_bytes: ]

            ctr_2d_ft += 1
    # =================== Split the intermediate bitstream ================== #

    # ========== Reconstruct some information from the header data ========== #
    # We use a mask with n_ctx_rowcol pixel left, right, up and down to obtain
    # the spatial neighbors.
    # If we need 3 rows & columns of context, we'll use a 7x7 mask as:
    #   1 1 1 1 1 1 1
    #   1 1 1 1 1 1 1
    #   1 1 1 1 1 1 1
    #   1 1 1 * 0 0 0
    #   0 0 0 0 0 0 0
    #   0 0 0 0 0 0 0
    #   0 0 0 0 0 0 0
    mask_size = 2 * header_info.get('n_ctx_rowcol') + 1
    # How many padding pixels we need
    pad = header_info.get('n_ctx_rowcol')
    # Number of context pixel used.
    non_zero_pixel_ctx = int((mask_size ** 2 - 1) / 2)
    # ========== Reconstruct some information from the header data ========== #

    # =========================== Decode the NNs ============================ #
    # To decode each NN (arm, upsampling and synthesis):
    #   1. Instantiate an empty Module
    #   2. Populate it with the weights and biases decoded from the bitstream
    #   3. Send it to the requested device.

    arm = decode_network(
        Arm(non_zero_pixel_ctx, header_info.get('layers_arm'), FIXED_POINT_FRACTIONAL_MULT),  # Empty module
        DescriptorNN(
            weight = f'{bitstream_path}_arm_weight',
            bias = f'{bitstream_path}_arm_bias',
        ),
        DescriptorNN (
            weight = POSSIBLE_Q_STEP_ARM_NN[header_info['q_step_index_nn']['arm']['weight']],
            bias = POSSIBLE_Q_STEP_ARM_NN[header_info['q_step_index_nn']['arm']['bias']],
        ),
        DescriptorNN (
            weight = POSSIBLE_SCALE_NN[header_info['scale_index_nn']['arm']['weight']],
            bias = POSSIBLE_SCALE_NN[header_info['scale_index_nn']['arm']['bias']],
        ),
        header_info.get('ac_max_val_nn'),
    )
    # Set the desired quantization accuracy for the ARM
    arm.set_quant(FIXED_POINT_FRACTIONAL_MULT)

    have_bias = header_info['q_step_index_nn']['upsampling']['bias'] >= 0
    if have_bias:
        # For the moment we do not expect this!  No bias for upsampling.
        print("WHAT")
        exit(1)
    upsampling = decode_network(
        Upsampling(
            header_info.get('upsampling_kernel_size')
        ),  # Empty module
        DescriptorNN(
            weight = f'{bitstream_path}_upsampling_weight',
            bias = "",
        ),
        DescriptorNN (
            weight = POSSIBLE_Q_STEP_SYN_NN[header_info['q_step_index_nn']['upsampling']['weight']],
            bias = 0,
        ),
        DescriptorNN (
            weight = POSSIBLE_SCALE_NN[header_info['scale_index_nn']['upsampling']['weight']],
            bias = 0,
        ),
        header_info.get('ac_max_val_nn'),
    )

    synthesis = decode_network(
        Synthesis(
            sum(header_info.get('n_ft_per_latent')), header_info.get('layers_synthesis')
        ),  # Empty module
        DescriptorNN(
            weight = f'{bitstream_path}_synthesis_weight',
            bias = f'{bitstream_path}_synthesis_bias',
        ),
        DescriptorNN (
            weight = POSSIBLE_Q_STEP_SYN_NN[header_info['q_step_index_nn']['synthesis']['weight']],
            bias = POSSIBLE_Q_STEP_SYN_NN[header_info['q_step_index_nn']['synthesis']['bias']],
        ),
        DescriptorNN (
            weight = POSSIBLE_SCALE_NN[header_info['scale_index_nn']['synthesis']['weight']],
            bias = POSSIBLE_SCALE_NN[header_info['scale_index_nn']['synthesis']['bias']],
        ),
        header_info.get('ac_max_val_nn'),
    )
    # =========================== Decode the NNs ============================ #

    # Instantiate a range coder to decode the y
    range_coder = RangeCoder(header_info.get('n_ctx_rowcol'), header_info.get('ac_max_val_latent'))
    decoded_y = []

    # Decode the different latent grids one after the other
    ctr_2d_ft = 0
    for index_lat_resolution in range(header_info.get('latent_n_resolutions')):
        h_grid, w_grid = [
            int(math.ceil(x / (2 ** index_lat_resolution)))
            for x in img_size
        ]

        c_grid = header_info.get('n_ft_per_latent')[index_lat_resolution]
        # Nothing to read, still increment the counter
        if header_info.get('n_bytes_per_latent')[ctr_2d_ft] == 0:
            current_y = torch.zeros((1, c_grid, h_grid, w_grid))
            decoded_y.append(current_y)
            ctr_2d_ft += 1
            continue

        for index_lat_feature in range(c_grid):
            # Nothing to read, still increment the counter
            if header_info.get('n_bytes_per_latent')[ctr_2d_ft] == 0:
                current_y = torch.zeros((1, c_grid, h_grid, w_grid))
                decoded_y.append(current_y)
                ctr_2d_ft += 1
                continue

            cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)

            range_coder.load_bitstream(cur_latent_bitstream)
            coding_order = range_coder.generate_coding_order(
                (1, h_grid, w_grid), header_info.get('n_ctx_rowcol')
            )

            # --------- Wave front coding order
            # With n_ctx_rowcol = 1, coding order is something like
            # 0 1 2 3 4 5 6
            # 1 2 3 4 5 6 7
            # 2 3 4 5 6 7 8
            # which indicates the order of the wavefront coding process.
            flat_coding_order = coding_order.flatten().contiguous()

            # flat_index_coding_is an array with [H_grid * W_grid] elements.
            # It indicates the index of the different coding cycle i.e.
            # the concatenation of the following list:
            #   [indices for which flat_coding_order == 0]
            #   [indices for which flat_coding_order == 1]
            #   [indices for which flat_coding_order == 2]
            flat_index_coding_order = flat_coding_order.argsort(stable=True)

            # occurrence_coding_order gives us the number of values to be decoded
            # at each wave-front cycle i.e. the number of values whose coding
            # order is i.
            # occurrence_coding_order[i] = number of values to decode at step i
            _, occurrence_coding_order = torch.unique(flat_coding_order, return_counts=True)

            # Current latent grid without padding
            current_y = torch.zeros((1, 1, h_grid, w_grid))

            # Compute the 1d offset of indices to form the context
            offset_index_arm = compute_offset(current_y, mask_size)

            # pad and flatten and current y to perform everything in 1d
            current_y = F.pad(current_y, (pad, pad, pad, pad), mode='constant', value=0.)
            current_y = current_y.flatten().contiguous()

            # Pad the coding order with unreachable values (< 0) to mimic the current_y padding
            # then flatten it as we want to index current_y which is a 1d tensor
            coding_order = F.pad(coding_order, (pad, pad, pad, pad), mode='constant', value=-1)

            coding_order = coding_order.flatten().contiguous()

            # Count the number of decoded values to iterate within the
            # flat_index_coding_order array.
            cnt = 0
            n_ctx_row_col = header_info.get('n_ctx_rowcol')
            for index_coding in range(flat_coding_order.max() + 1):
                cur_context = fast_get_neighbor(
                    current_y, mask_size, offset_index_arm,
                    flat_index_coding_order[cnt: cnt + occurrence_coding_order[index_coding]],
                    w_grid
                )

                # Compute proba param from context
                if ARMINT:
                    # Run on CPU if ARM is done with actual integer (and not integer-in-float mode)
                    cur_raw_proba_param = arm(cur_context.cpu())
                else:
                    cur_raw_proba_param = arm(cur_context)
                cur_mu, cur_scale = get_mu_scale(cur_raw_proba_param)

                # Decode and store the value at the proper location within current_y
                x = range_coder.decode(cur_mu.cpu(), cur_scale.cpu())
                x_delta = n_ctx_row_col + 1
                if index_coding < w_grid:
                    start_y = 0
                    start_x = index_coding
                elif True:
                    start_y = (index_coding-w_grid)//x_delta+1
                    start_x = w_grid - x_delta + (index_coding-w_grid)%x_delta

                current_y[
                    [
                        (w_grid + 2 * pad) * (pad + start_y + i) + (pad + start_x - x_delta * i)
                        for i in range(len(x))
                    ]
                ] = x
                # current_y[coding_order == index_coding] = range_coder.decode(cur_mu, cur_scale)

                # Increment the counter of loaded value
                cnt += occurrence_coding_order[index_coding]

            # Reshape y as a 4D grid, and remove padding
            current_y = current_y.reshape(1, 1, h_grid + 2 * pad, w_grid + 2 * pad)
            current_y = current_y[:, :, pad:-pad, pad:-pad]
            decoded_y.append(current_y)

            ctr_2d_ft += 1

    # Up until there, decoded_y is a list of 2d tensors, group them as a (smaller) list of 3d
    final_decoded_y = []
    ctr_2d_ft = 0
    for index_lat_resolution in range(header_info.get('latent_n_resolutions')):
        c_grid = header_info.get('n_ft_per_latent')[index_lat_resolution]
        h_grid, w_grid = [
            int(math.ceil(x / (2 ** index_lat_resolution)))
            for x in img_size
        ]

        if c_grid == 0:
            final_decoded_y.append(torch.zeros((1, 0, h_grid, w_grid)))
        else:
            final_decoded_y.append(
                torch.cat(decoded_y[ctr_2d_ft: ctr_2d_ft + c_grid], dim=1)
            )
        ctr_2d_ft += max(c_grid, 1)

    # Upsample the latents
    synthesis_input = upsampling(final_decoded_y)

    # Reconstruct the output
    synthesis_output = synthesis(synthesis_input)

    # ================= Clean up the intermediate bitstream ================= #
    for cur_module_name in ['arm', 'synthesis', 'upsampling']:
        for parameter_type in ['weight', 'bias']:
            cur_bitstream = f'{bitstream_path}_{cur_module_name}_{parameter_type}'
            subprocess.call(f'rm -f {cur_bitstream}', shell=True)

    ctr_2d_ft = 0
    for index_lat_resolution in range(header_info.get('latent_n_resolutions')):
        # No feature map... we still increment the counter and delete the subbitstream file
        if header_info.get('n_ft_per_latent')[index_lat_resolution] == 0:
            cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
            subprocess.call(f'rm -f {cur_latent_bitstream}', shell=True)
            ctr_2d_ft += 1

        for index_lat_feature in range(header_info.get('n_ft_per_latent')[index_lat_resolution]):
                cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
                subprocess.call(f'rm -f {cur_latent_bitstream}', shell=True)
                ctr_2d_ft += 1
    # ================= Clean up the intermediate bitstream ================= #

    elapsed = time.time() - start_time
    print(f'Frame decoding time: {elapsed:4.3f} sec')

    return synthesis_output, bitstream

@torch.no_grad()
def decode_video(bitstream_path: str, output_path: str):
    start_time = time.time()
    torch.use_deterministic_algorithms(True)

    # ========================== Parse the header =========================== #
    header_info = read_gop_header(bitstream_path)
    # ========================== Parse the header =========================== #

    # Read all the bits and discard the header
    with open(bitstream_path, 'rb') as fin:
        bitstream = fin.read()[header_info.get('n_bytes_header'): ]

    coding_structure = CodingStructure(
        gop_type=header_info.get('gop_type'),
        intra_period=header_info.get('intra_period'),
        p_period=header_info.get('p_period'),
    )


    for idx_coding_order in range(coding_structure.get_number_of_frames()):
        frame = coding_structure.get_frame_from_coding_order(idx_coding_order)

        # ================ Load the references from already decoded images ================ #
        ref_data = []
        for idx_ref in frame.index_references:
            ref_frame = coding_structure.get_frame_from_display_order(idx_ref)

            # 4:2:0 Reference are upsampled before being used by the InterCodingModule
            if header_info.get('frame_data_type') == 'yuv420':
                decoded_frame = convert_420_to_444(ref_frame.decoded_data.data)
                ref_frame_data_type = 'yuv444'
            else:
                decoded_frame = ref_frame.decoded_data.data
                ref_frame_data_type = header_info.get('frame_data_type')
            ref_data.append(FrameData(header_info.get('bitdepth'), ref_frame_data_type, decoded_frame))

        frame.set_refs_data(ref_data)
        # ================ Load the references from already decoded images ================ #

        # Forward into the cool chic decoder... with the ARM module.
        # We also return the bitstream with the read pointer updated?
        raw_coolchic_output, bitstream = decode_frame(bitstream_path, bitstream, img_size=header_info.get('img_size'))

        # Feed the output of coolchic to the inter coding module
        inter_coding_module = InterCodingModule(frame_type=frame.frame_type)
        # Set the flow gain
        if len(frame.index_references) > 0:
            inter_coding_module.flow_gain = abs(frame.display_order - frame.index_references[0])

        decoded_image = inter_coding_module.forward(
            {
                'raw_out': raw_coolchic_output,
                # Dummy values to match with the expected signature
                'rate': torch.zeros((1)),
                'additional_data': {}
            },
            references=[frame_data.data for frame_data in frame.refs_data]
        ).decoded_image

        # Round the output according to the bitdepth
        # Then scale it back into a "discrete" tensor in [0, 1]
        bitdepth = header_info.get('bitdepth')
        decoded_image = torch.round((2 ** bitdepth - 1) * decoded_image)  / (2 ** bitdepth - 1)

        if 'yuv420' in header_info.get('frame_data_type'):
            decoded_image = yuv_dict_clamp(convert_444_to_420(decoded_image), 0., 1.)
        else:
            decoded_image = torch.clamp(decoded_image, 0., 1.)

        frame.set_decoded_data(
            FrameData(
                bitdepth=header_info.get('bitdepth'),
                frame_data_type=header_info.get('frame_data_type'),
                data=decoded_image,
            )
        )

        # Save the currently decoded frame, we'll concatenate them later
        save_visualisation(
            frame.decoded_data,
            f'{output_path}_{frame.display_order}',
            mode='image',
            format='png' if header_info.get('frame_data_type') == 'rgb' else 'yuv'
        )

    # Delete the output file if it exists
    format = 'png' if header_info.get('frame_data_type') == 'rgb' else 'yuv'
    subprocess.call(f'rm -f {output_path}', shell=True)

    for idx_display_order in range(coding_structure.get_number_of_frames()):
        # The save_visualisation function will add some extra character in the
        # name of the file (e.g. HxW_bitdepth_yuv420.yuv). Some we need to glob
        # with the beginning of the name to try finding the corresponding frame.
        frame_path = glob.glob(f'{output_path}_{idx_display_order}*')
        assert len(frame_path) == 1, f'Found {len(frame_path)} corresponding frame(s) ' \
            f'for the frame n° {idx_display_order}. Should have been 1.'

        frame_path = frame_path[0]
        subprocess.call(f'cat {frame_path} >> {output_path}', shell=True)
        subprocess.call(f'rm -f {frame_path}', shell=True)

    elapsed = time.time() - start_time
    print(f'Total decoding time: {elapsed:4.3f} sec')
