# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

# Some code here is inspired from
# https://github.com/microsoft/DCVC/blob/main/DCVC-FM/src/models/block_mc.py
# License: MIT

import torch
from torch import Tensor, nn


# backward_grid = None

def warp_fn(x: Tensor, flow: Tensor) -> Tensor:
    """Motion compensation (warping) of a tensor [B, C, H, W] with a 2-d displacement
    [B, 2, H, W].

    Args:
        x (Tensor): Tensor to be motion compensated [B, C, H, W].
        flow (Tensor): Displacement [B, C, H, W]. flow[:, 0, :, :] corresponds to
            the horizontal displacement. flow[:, 1, :, :] is the vertical displacement.

    Returns:
        Tensor: Motion compensated tensor [B, C, H, W].
    """
    B, _, H, W = x.size()
    cur_device = x.device

    # TODO: Could be better managed by avoiding the reallocation each time
    # TODO: we call the warp function
    # global backward_grid
    # if backward_grid is None:
    tensor_hor = (
        torch.linspace(-1.0, 1.0, W, device=cur_device, dtype=torch.float32)
        .view(1, 1, 1, W)
        .expand(B, -1, H, -1)
    )
    tensor_ver = (
        torch.linspace(-1.0, 1.0, H, device=cur_device, dtype=torch.float32)
        .view(1, 1, H, 1)
        .expand(B, -1, -1, W)
    )
    backward_grid = torch.cat([tensor_hor, tensor_ver], 1)

    flow = torch.cat(
        [
            flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
            flow[:, 1:2, :, :] / ((H - 1.0) / 2.0),
        ],
        dim=1,
    )

    grid = backward_grid + flow

    output = nn.functional.grid_sample(
        x,
        grid.permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return output
