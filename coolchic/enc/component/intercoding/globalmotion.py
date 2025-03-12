# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from typing import List, Tuple

import torch
from enc.component.intercoding.warp import warp_fn
from enc.io.format.yuv import convert_420_to_444, convert_444_to_420
from enc.io.framedata import FrameData
from enc.training.loss import _compute_mse
from torch import Tensor


@torch.no_grad()
def get_global_translation(
    frame_data: FrameData, ref_data: List[FrameData]
) -> Tuple[List[Tensor], List[Tensor]]:
    """This function computes the best global translation for each reference.
    The best translation is a 2-element vector :math:`\\mathbf{v} = (v_x, x_y)`
    which maximises the PSNR of a reference warping with the frame to code:

    .. math ::

        \\mathbf{v}^\\star = \\arg\\min = ||\\mathrm{warp}(\\hat{\\mathbf{x}}_{ref},
        \\mathbf{v}) - \\mathbf{x} ||_2^2, \\text{ with }
        \\begin{cases}
            \\mathbf{x} & \\text{the original image}\\\\
            \\hat{\\mathbf{x}}_{ref} & \\text{a reference frame}
        \\end{cases}

    Since they can be multiple references, we look for the best global
    translation for each reference.

    Beside the global translation, also return the shifted reference (by the
    aforementioned global translation). They are required to feed an optical
    flow estimator afterward.

    Args:
        frame_data: Data representing the frame to be encoded.
        ref_data: Data representing the reference(s)

    Returns:
        Tuple[List[Tensor], List[Tensor]]: First element is the list of the
            shifted references. Second element is the list of the global
            translations.
    """

    print("Global translation estimation")
    print("-----------------------------")

    updated_ref = []
    global_flow = []
    for i, ref_data_i in enumerate(ref_data):
        print(f"\nReference {i}")
        raw_ref_data = ref_data_i.data
        raw_target_data = frame_data.data

        # RAFT expects dense frame, not 420 format.
        if ref_data_i.frame_data_type == "yuv420":
            raw_ref_data = convert_420_to_444(raw_ref_data)
        if frame_data.frame_data_type == "yuv420":
            raw_target_data = convert_420_to_444(raw_target_data)

        device = raw_ref_data.device
        h, w = raw_ref_data.size()[-2:]
        center_flow = torch.zeros((1, 2, h, w), device=device)

        # Use a coarse to find approach. Given a starting position center_flow,
        # we test all the possible combination of flow_x = (-step_size, 0, step_size)
        # and flow_y = (-step_size, 0, step_size). Then we keep the best one as
        # our new center, we divide step_size and we keep doing that until a small
        # enough step_size (here step_size2) is reached.
        for n in range(8, 0, -1):
            shift_size = 2**n

            all_shifts = [
                torch.tensor([0, 0]),
                torch.tensor([0, shift_size]),
                torch.tensor([0, -shift_size]),
                torch.tensor([shift_size, 0]),
                torch.tensor([-shift_size, 0]),
                torch.tensor([shift_size, shift_size]),
                torch.tensor([-shift_size, shift_size]),
                torch.tensor([shift_size, -shift_size]),
                torch.tensor([-shift_size, -shift_size]),
            ]
            all_shifts = map(lambda x: x.to(device).view((1, 2, 1, 1)), all_shifts)

            initial_mse = _compute_mse(raw_ref_data, raw_target_data)
            initial_shift = torch.tensor([0, 0]).to(device).view((1, 2, 1, 1))
            best_mse = initial_mse
            best_shift = initial_shift
            best_psnr = -10 * torch.log10(best_mse + 1e-10)

            for shift in all_shifts:
                # All flows here are a constant [1, 2, H, W] tensor.
                flow = torch.ones((1, 2, h, w), device=device) * shift + center_flow

                shifted_ref = warp_fn(raw_ref_data, flow)
                shifted_mse = _compute_mse(shifted_ref, raw_target_data)
                shifted_psnr = -10 * torch.log10(shifted_mse + 1e-10)

                if shifted_mse < best_mse:
                    best_shift = shift
                    best_mse = shifted_mse
                    best_psnr = -10 * torch.log10(best_mse + 1e-10)

            center_flow += best_shift
            shifted_ref = warp_fn(raw_ref_data, center_flow)
            shifted_mse = _compute_mse(shifted_ref, raw_target_data)
            shifted_psnr = -10 * torch.log10(shifted_mse + 1e-10)
            print(
                f"Search size: +/- {shift_size:<4} "
                f"Current translation: {center_flow[0, 0, 0, 0]:>6};{center_flow[0, 1, 0, 0]:>6} ; "
                f"PSNR = {shifted_psnr:6.3f} dB"
            )

        initial_psnr = -10 * torch.log10(initial_mse + 1e-10)
        print(
            f"\nReference {i} | "
            f"Final global translation: ({center_flow[0, 0, 0, 0]:}, {center_flow[0, 1, 0, 0]:})"
        )
        print(
            f"Reference {i} | "
            f"Initial PSNR(ref, to code) = {initial_psnr:6.3f} dB | "
            f"Final PSNR(shifted ref, to code) = {best_psnr:6.3f} dB"
        )

        if ref_data_i.frame_data_type == "yuv420":
            shifted_ref = convert_444_to_420(shifted_ref)

        updated_ref.append(
            FrameData(
                bitdepth=ref_data_i.bitdepth,
                frame_data_type=ref_data_i.frame_data_type,
                data=shifted_ref,
            )
        )
        global_flow.append(center_flow[0, :, 0, 0])

    return updated_ref, global_flow
