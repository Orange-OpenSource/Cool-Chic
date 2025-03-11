# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from typing import List

import torch
from enc.component.intercoding.warp import warp_fn
from enc.io.format.yuv import convert_420_to_444, rgb2yuv, yuv2rgb
from enc.io.framedata import FrameData
from enc.training.loss import _compute_mse
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


@torch.no_grad()
def get_raft_optical_flow(
    frame_data: FrameData, ref_data: List[FrameData]
) -> List[torch.Tensor]:
    """Apply a RAFT model on its frame and each of its reference.
    Return the raft-estimated optical flows.

    Args:
        frame_data: Data for the frame to be encoded.
        ref_data: Data of the references.

    Returns:
        List[Tensor]: List of optical flows. Each of them with a shape of [1, 2, H, W].
    """

    print("\nRaft-based motion estimation")
    print("----------------------------\n")

    # Store the old device to move the flows etc. after the RAFT
    if frame_data.frame_data_type == "yuv420":
        old_device = frame_data.data.get("y").device
    else:
        old_device = frame_data.data.device

    if torch.cuda.is_available():
        total_gpu_mem_gbytes = torch.cuda.mem_get_info()[1] / 1e9
        _RAFT_DEVICE = "cpu" if total_gpu_mem_gbytes < 24 else "cuda:0"
    else:
        _RAFT_DEVICE = "cpu"

    # Get a pre-trained raft model to guide the coolchic encoder training
    # with a good optical flow
    raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
    raft = raft.to(_RAFT_DEVICE)
    raft.eval()

    flows = []
    for i, ref_data_i in enumerate(ref_data):
        raw_ref_data = ref_data_i.data
        raw_target_data = frame_data.data

        # RAFT expects dense frame, not 420 format.
        if ref_data_i.frame_data_type == "yuv420":
            raw_ref_data = convert_420_to_444(raw_ref_data)
        if frame_data.frame_data_type == "yuv420":
            raw_target_data = convert_420_to_444(raw_target_data)

        # RAFT expects RGB data in [-1., 1.]
        if ref_data_i.frame_data_type != "rgb":
            raw_ref_data = yuv2rgb(raw_ref_data * 255.0) / 255.0
        if frame_data.frame_data_type != "rgb":
            raw_target_data = yuv2rgb(raw_target_data * 255.0) / 255.0

        # Convert the data from [0, 1] -> [-1, 1]
        raw_target_data = raw_target_data * 2 - 1
        raw_ref_data = raw_ref_data * 2 - 1

        # Send from old_device -> _RAFT_DEVICE prior to using raft.
        # After raft, send back from _RAFT_DEVICE -> old_device.
        raw_target_data = raw_target_data.to(_RAFT_DEVICE)
        raw_ref_data = raw_ref_data.to(_RAFT_DEVICE)
        raft_flow = raft(raw_target_data, raw_ref_data)[-1]
        raw_target_data = raw_target_data.to(old_device)
        raw_ref_data = raw_ref_data.to(old_device)
        raft_flow = raft_flow.to(old_device)

        # Send back the data from [-1, 1] -> [0, 1]
        raw_target_data = (raw_target_data + 1) * 0.5
        raw_ref_data = (raw_ref_data + 1) * 0.5

        flows.append(raft_flow)
        raft_pred = warp_fn(raw_ref_data, raft_flow)

        # Compute the PSNR of the prediction obtained by raft.
        # Note that we compute this prediction in the 444 domain even though
        # we have a 420 frame.
        raft_pred_psnr = -10 * torch.log10(
            _compute_mse(
                rgb2yuv(raft_pred * 255.0) / 255.0,
                rgb2yuv(raw_target_data * 255.0) / 255.0,
            )
            + 1e-10
        )

        # Just as a reference we compute the PSNR when the pred is the
        # reference i.e. optical flow = 0.
        dummy_pred_psnr = -10 * torch.log10(
            _compute_mse(
                rgb2yuv(raw_ref_data * 255.0) / 255.0,
                rgb2yuv(raw_target_data * 255.0) / 255.0,
            )
            + 1e-10
        )

        print(
            f"Reference {i} | "
            f"RAFT-based prediction PSNR = {raft_pred_psnr:6.3f} dB | "
            f"No motion prediction PSNR = {dummy_pred_psnr:6.3f} dB"
        )

    return flows
