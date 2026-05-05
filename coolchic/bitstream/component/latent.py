from typing import List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from coolchic.bitstream.component.armint import MU_LOG_SCALE_MIN_FIXED_POINT, fixed_point_arm
from coolchic.bitstream.component.constants import (
    FIXED_POINT_DTYPE,
    N_FRAC_BIT_MU_SCALE,
    WEIGHT_SHIFT,
)
from coolchic.bitstream.component.rangecoder import RangeCoder
from coolchic.component.core.arm import _get_mask_size_ctx, _get_non_zero_pixel_ctx_index


def entropy_coding_latent_arm(
    encoder_data: Optional[Tensor],  # Data is [1, 1, H, W]
    context_inter_features: Optional[Tensor],  # [1, C, H, W]
    spatial_dim: Tuple[int, int],
    fixed_point_weights: List[Tensor],
    fixed_point_bias: List[Tensor],
    fixed_point_stab_weights: Tensor,
    fixed_point_stab_biases: Tensor,
    range_coder: RangeCoder,
    mode: Literal["encode", "decode"],
    n_spatial_context: int,
) -> Tensor:
    """Either encode the encoder data into the range_coder internal byte buffer (if mode=="encode")
    or decoder the range coder byte buffer.

    For the encode mode, use range_coder.get_bitstream_bytes() to obtain the bytes ready to
    be saved into the bitstream.

    Args:
        encoder_data (Optional[Tensor]): Latent to encode. Shape is [1, 1, H, W].
            None if mode == "decode"
        context_inter_features (Optional[Tensor]): IFCE output, already computed from previously
            decoded latent grids. Shape is [1, C, H, W]
        fixed_point_weights (List[Tensor]): List of 2D tensors representing the weights for each
            layer of the main trunks.
        fixed_point_bias (List[Tensor]): List of 1D tensors representing the biases for each
            layer of the main trunks.
        fixed_point_stab_weights (Tensor): 2D weight Tensor of the stabiliser branch. Always present,
            even if no stabiliser branch. In this case, it is the Identity matrix.
        fixed_point_stab_biases (Tensor): 1D bias Tensor of the stabiliser branch. Always present,
            even if no stabiliser branch. In this case, it is an all-zero vector.
        range_coder (RangeCoder): Range Coder object used to perform the entropy coding
        mode (Literal["encode", "decode"]): _description_
        n_spatial_context (int): Number of spatial contexts.

    Returns:
        Tensor: The decoded latent of shape [1, 1, H, W]. Should be equal to encoder_data when mode == "encode"
    """

    if encoder_data is not None:
        if torch.is_floating_point(encoder_data) or torch.is_complex(encoder_data):
            raise TypeError(
                f"Entropy coded latent should be integer. Found dtype={encoder_data.dtype}"
            )

    ARM_MASK_SIZE = _get_mask_size_ctx()

    h, w = spatial_dim
    # --------- Wave front coding order
    # With ARM_MASK_SIZE = 1, coding order is something like
    # 0 1 2 3 4 5 6
    # 1 2 3 4 5 6 7
    # 2 3 4 5 6 7 8
    # which indicates the order of the wavefront coding process.
    coding_order = generate_coding_order((1, h, w), ARM_MASK_SIZE)

    # occurrence_coding_order gives us the number of values to be decoded
    # at each wave-front cycle i.e. the number of values whose coding
    # order is i.
    # occurrence_coding_order[i] = number of values to decode at step i
    _, occurrence_coding_order = torch.unique(coding_order, return_counts=True)
    max_parallel_decoded = occurrence_coding_order.max()

    # Compute the 1d offset of indices to form the context
    offset_index_arm = compute_offset(
        spatial_dim, ARM_MASK_SIZE, _get_non_zero_pixel_ctx_index(n_spatial_context)
    )

    # Padding so that we can have an entire (all-zero) context for the top left pixel
    pad = (ARM_MASK_SIZE - 1) // 2
    padded_width = w + 2 * pad

    if encoder_data is not None:
        # pad and flatten the current data to perform everything in 1d
        encoder_data = F.pad(encoder_data, (pad, pad, pad, pad), mode="constant", value=0.0)
        encoder_data = encoder_data.flatten().contiguous()

    if context_inter_features is not None:
        context_inter_features = F.pad(
            context_inter_features, (pad, pad, pad, pad), mode="constant", value=0.0
        )
        context_inter_features = rearrange(context_inter_features, "1 c h w -> (h w) c")

    data_to_fill = torch.zeros((1, 1, h + 2 * pad, w + 2 * pad), dtype=FIXED_POINT_DTYPE)
    data_to_fill = data_to_fill.flatten().contiguous()

    # Pad the coding order with unreachable values (< 0) to mimic the current_y padding
    # then flatten it as we want to index current_y which is a 1d tensor
    coding_order = F.pad(coding_order, (pad, pad, pad, pad), mode="constant", value=-1)
    coding_order = coding_order.flatten().contiguous()

    # +1 because we do want to include the last value in coding_order.max()
    all_index_coding = torch.arange(coding_order.max() + 1, dtype=torch.int32)

    # No wavefront for line shorter than the (maximum) arm template
    if w <= ARM_MASK_SIZE:
        all_idx = [
            (
                (pad + row_idx) * (w + 2 * pad)  # Skip first rows
                + pad  # Skip padded cols of first non-empty row
                + torch.arange(w)  # Actual pixel to code
            )
            for row_idx in range(h)
        ]
        all_idx = torch.cat(all_idx).view(-1, 1)
    else:
        # Compute the wave front indices
        all_start_y = torch.zeros_like(all_index_coding, dtype=torch.int32)
        all_start_x = torch.arange(coding_order.max() + 1, dtype=torch.int32)
        all_start_y[w:] = (all_index_coding[w:] - w) // (ARM_MASK_SIZE + 1) + 1
        all_start_x[w:] = w - (ARM_MASK_SIZE + 1) + (all_index_coding[w:] - w) % (ARM_MASK_SIZE + 1)

        i = torch.arange(max_parallel_decoded, dtype=torch.int32).view(1, -1)
        all_start_x_repeat = all_start_x.view(-1, 1).repeat(1, max_parallel_decoded)
        all_start_y_repeat = all_start_y.view(-1, 1).repeat(1, max_parallel_decoded)

        all_idx = padded_width * (pad + all_start_y_repeat + i) + (
            pad + all_start_x_repeat - (ARM_MASK_SIZE + 1) * i
        )

    all_neighbor_idx = (
        all_idx.view(-1, max_parallel_decoded, 1).repeat(1, 1, n_spatial_context) - offset_index_arm
    )

    for index_coding in range(coding_order.max() + 1):
        n_decoded_value = occurrence_coding_order[index_coding]
        idx = all_idx[index_coding, :n_decoded_value]

        neighbor_idx = all_neighbor_idx[index_coding, :n_decoded_value, :].flatten()
        # Pick context in data to fill to avoid seing forbidden value
        context = torch.index_select(data_to_fill, dim=0, index=neighbor_idx).view(
            -1, n_spatial_context
        )

        if context_inter_features is not None:
            context = torch.cat((context, context_inter_features[idx, :]), dim=1)

        # Compute proba param from context
        mu_scale = fixed_point_arm(
            context,
            fixed_point_weights,
            fixed_point_bias,
            fixed_point_stab_weights,
            fixed_point_stab_biases,
            output_shift=2 * WEIGHT_SHIFT - N_FRAC_BIT_MU_SCALE,
        )
        # Subtract the minimum available values for mu and sigma XXX_MIN_FIXED_POINT
        idx_mu_scale = mu_scale - MU_LOG_SCALE_MIN_FIXED_POINT

        # Decode and store the value at the proper location within current_y
        if mode == "encode":
            range_coder.encode(encoder_data[idx], idx_mu_scale)
            data_to_fill[idx] = encoder_data[idx]

        elif mode == "decode":
            data_to_fill[idx] = range_coder.decode(idx_mu_scale).to(FIXED_POINT_DTYPE)

    if mode == "encode":
        if encoder_data.not_equal(data_to_fill).any():
            raise ValueError(
                "Error when encoding latent. Actually encoded values is not equal "
                "to the expected encoded values. "
                f"a = {encoder_data.abs().sum()}; b = {data_to_fill.abs().sum()}"
            )

    # Reshape y as a 4D grid, and remove padding
    data_to_fill = data_to_fill.reshape(1, 1, h + 2 * pad, w + 2 * pad)
    data_to_fill = data_to_fill[:, :, pad:-pad, pad:-pad]

    return data_to_fill


def compute_offset(
    spatial_dim: Tuple[int, int], mask_size: int, non_zero_pixel_ctx_index: Tensor
) -> Tensor:
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
        mask_size (int): Size of the mask in which the context is contained

    Returns:
        Tensor: The offset vector
    """
    # Padding required by the number of context pixels
    pad = int((mask_size - 1) / 2)
    # # Number of input for the ARM MLP
    # n_context_pixel = int((mask_size ** 2 - 1) / 2)
    # n_context_pixel = non_zero_pixel_ctx_index.numel()

    # Size of the actual and padded feature maps.
    H, W = spatial_dim
    W_pad = W + 2 * pad

    idx_row = pad - non_zero_pixel_ctx_index // mask_size
    idx_col = pad - non_zero_pixel_ctx_index % mask_size

    # 1d offset. Moving one row up shifts the 1d index by W_pad pixels
    offset = idx_col + idx_row * W_pad
    return offset


def generate_coding_order(CHW: Tuple[int, int, int], arm_mask_size: int) -> Tensor:
    C, H, W = CHW

    # Edge case: we have an image whose width is smaller than the number of
    # context row/column in this case: no wavefront
    if W <= arm_mask_size:
        coding_order = torch.arange(0, H * W).view(1, H, W).repeat(C, 1, 1).view(1, C, H, W)
        return coding_order

    # Generate something like
    # 0                 1                   2                  3
    # N + 1             N + 2               N + 3              N + 4
    # 2 * (N + 1)       2 * (N + 1) + 1     2 * (N + 1) + 2    2 * (N + 1) + 3
    # with N the number of row/column of context

    # Repeat the first line H times
    first_line = torch.arange(W).view(1, -1).repeat((H, 1))
    row_increment = torch.arange(H) * (arm_mask_size + 1)
    row_increment = row_increment.view(-1, 1)

    # This is the spatial coding order i.e. a [H, W] tensor
    # since we code the C channels in parallel we just have to repeat
    # the spatial coding order along each channels
    coding_order = first_line + row_increment
    coding_order = coding_order.view(1, H, W).repeat(C, 1, 1).view(1, C, H, W)
    return coding_order
