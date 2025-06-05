# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

from enc.component.coolchic import CoolChicEncoderOutput
from enc.utils.codingstructure import FRAME_TYPE


def warp(x, flo, interpol_mode="bilinear", padding_mode="border", align_corners=True):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    cur_device = x.device

    # mesh grid
    xx = torch.arange(0, W, device=cur_device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=cur_device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    # Already on cur_device as xx and yy are created on cur_device
    grid = torch.cat((xx, yy), 1).float()

    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(
        x,
        vgrid,
        mode=interpol_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    mask = torch.autograd.Variable(torch.ones(x.size(), device=cur_device))
    mask = nn.functional.grid_sample(
        mask,
        vgrid,
        mode=interpol_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


@dataclass
class InterCodingModuleInput:
    """Represent the data required by the InterCodingModule. It is obtained by processing
    the raw CoolChicEncoderOutput."""

    residue: Optional[Tensor] = (
        None  # [B, 3, H, W] either the residue or the entire frame for I frames
    )
    flow_1: Optional[Tensor] = (
        None  # [B, 2, H, W] optical flow to interpolate the 1st reference. Zero for I frames
    )
    flow_2: Optional[Tensor] = (
        None  # [B, 2, H, W] optical flow to interpolate the 2nd reference. Zero for I & P frames
    )
    alpha: Optional[Tensor] = (
        None  # [B, 1, H, W] Intra / inter switch. Zero for I frames
    )
    beta: Optional[Tensor] = (
        None  # [B, 1, H, W] Bi-directional prediction weighting. One for I & P frames
    )


@dataclass
class InterCodingModuleOutput:
    """Represent the output of the InterCodingModule."""

    decoded_image: Tensor  # [B, 3, H, W] Decoded image. Must be quantized on 10 or 8 bit by the VideoFrameEncoder

    # Any other data required to compute some logs, stored inside a dictionary
    additional_data: Dict[str, Any] = field(default_factory=lambda: {})


class InterCodingModule(nn.Module):
    def __init__(self, frame_type: FRAME_TYPE):
        """Initialize an InterCodingModule for a frame of type <frame_type>

        Args:
            frame_type (FRAME_TYPE): Either 'I', 'P' or 'B'
        """
        super().__init__()

        self.frame_type = frame_type

        # Multiply the optical flow coming from cool chic
        self.flow_gain = 1.0

    def process_coolchic_output(
        self, coolchic_output: CoolChicEncoderOutput
    ) -> InterCodingModuleInput:
        """Convert the raw, dense tensor from CoolChicEncoder forward into the different
        quantities expected by the InterCodingModule.

        Args:
            coolchic_output (CoolChicEncoderOutput): Dense [B, X, H, W] tensor from 

        Returns:
            InterCodingModuleInput: Data expected by the InterCodingModule
        """
        raw_coolchic_output = coolchic_output.get("raw_out")
        residue = raw_coolchic_output[:, :3, :, :]

        # b, _, h, w = residue.size()

        if self.frame_type == "P" or self.frame_type == "B":
            flow_1 = raw_coolchic_output[:, 3:5, :, :] * self.flow_gain
            alpha = torch.clamp(raw_coolchic_output[:, 5:6, :, :] + 0.5, 0.0, 1.0)
        else:
            flow_1 = None
            alpha = None

        if self.frame_type == "B":
            flow_2 = raw_coolchic_output[:, 6:8, :, :] * self.flow_gain
            beta = torch.clamp(raw_coolchic_output[:, 8:9, :, :] + 0.5, 0.0, 1.0)
        else:
            flow_2 = None
            beta = None

        return InterCodingModuleInput(
            residue=residue, flow_1=flow_1, flow_2=flow_2, alpha=alpha, beta=beta
        )

    def forward(
        self,
        coolchic_output: CoolChicEncoderOutput,
        references: List[Tensor],
        flag_additional_outputs: bool = False,
    ) -> InterCodingModuleOutput:
        """Process a raw CoolChic input and combine it with the references to obtain the
        decoded frame as a [B, 3, H, W] tensor.

        Args:
            coolchic_output (CoolChicEncoderOutput): Contains at least a key 'raw_out' with a [B, C, H, W]
                tensor describing the synthesis output.
            references (List[Tensor]): A list of 0 to 2 references as dense [B, 3, H, W] tensors.
            flag_additional_outputs (bool, optional): True to fill InterCodingModuleOutput.additional_data
                with many different quantities which can be used to analyze Cool-chic behavior.
                Defaults to False.

        Returns:
            InterCodingModuleOutput: Decoded frames among other things... see above for more details.
        """
        # Convert raw CoolChic output into meaningful quantities
        input_inter_coding = self.process_coolchic_output(coolchic_output)

        if self.frame_type == "I":
            decoded_frame = input_inter_coding.residue
        else:
            if self.frame_type == "P":
                # No beta: always set to one
                prediction = warp(references[0], input_inter_coding.flow_1)
            if self.frame_type == "B":
                # Prediction obtained by weighting two intermediate predictions
                prediction = input_inter_coding.beta * warp(
                    references[0], input_inter_coding.flow_1
                ) + (1 - input_inter_coding.beta) * warp(
                    references[1], input_inter_coding.flow_2
                )

            # Combine masked prediction & residue
            masked_prediction = input_inter_coding.alpha * prediction
            decoded_frame = masked_prediction + input_inter_coding.residue

        additional_data = {}
        if flag_additional_outputs:
            additional_data["residue"] = input_inter_coding.residue

            if self.frame_type == "P" or self.frame_type == "B":
                additional_data["alpha"] = input_inter_coding.alpha
                additional_data["flow_1"] = input_inter_coding.flow_1
                additional_data["prediction"] = prediction
                additional_data["masked_prediction"] = masked_prediction

            if self.frame_type == "B":
                additional_data["beta"] = input_inter_coding.beta
                additional_data["flow_2"] = input_inter_coding.flow_2

        return InterCodingModuleOutput(
            decoded_image=decoded_frame, additional_data=additional_data
        )
