# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


"""A frame encoder is composed of a CoolChicEncoder and a InterCodingModule."""

import typing
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, OrderedDict, Union

import torch
from enc.component.coolchic import (
    CoolChicEncoder,
    CoolChicEncoderParameter,
)
from enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
)
from enc.component.intercoding import InterCodingModule
from torch import Tensor, nn
from enc.utils.codingstructure import (
    FRAME_DATA_TYPE,
    FRAME_TYPE,
    POSSIBLE_BITDEPTH,
    DictTensorYUV,
    convert_444_to_420,
)
from enc.utils.misc import POSSIBLE_DEVICE
from enc.utils.yuv import yuv_dict_clamp


@dataclass
class FrameEncoderOutput:
    """Dataclass representing the output of FrameEncoder forward."""

    # Either a [B, 3, H, W] tensor representing the decoded image or a
    # dictionary with the following keys for yuv420:
    #   {
    #         'y': [B, 1, H, W],
    #         'u': [B, 1, H / 2, W / 2],
    #         'v': [B, 1, H / 2, W / 2],
    #   }
    # Note: yuv444 data are represented as a simple [B, 3, H, W] tensor
    decoded_image: Union[Tensor, DictTensorYUV]
    rate: Tensor  # Rate associated to each latent [total_latent_value]

    # Any other data required to compute some logs, stored inside a dictionary
    additional_data: Dict[str, Any] = field(default_factory=lambda: {})


class FrameEncoder(nn.Module):
    """A ``FrameEncoder`` is the object containing everything
    required to encode a video frame or an image. It is composed of
    a ``CoolChicEncoder`` and an ``ÌnterCodingModule``.
    """

    def __init__(
        self,
        coolchic_encoder_param: CoolChicEncoderParameter,
        frame_type: FRAME_TYPE = "I",
        frame_data_type: FRAME_DATA_TYPE = "rgb",
        bitdepth: POSSIBLE_BITDEPTH = 8,
    ):
        """
        Args:
            coolchic_encoder_param: Parameters for the underlying CoolChicEncoder
            frame_type: More info in
                :doc:`coding_structure.py <../utils/codingstructure>`.
                Defaults to "I".
            frame_data_type: More info in
                :doc:`coding_structure.py <../utils/codingstructure>`.
                Defaults to "rgb"
            bitdepth: More info in
                :doc:`coding_structure.py <../utils/codingstructure>`.
                Defaults to 8.
        """
        super().__init__()

        # ----- Copy the parameters
        self.coolchic_encoder_param = coolchic_encoder_param
        self.frame_type = frame_type
        self.frame_data_type = frame_data_type
        self.bitdepth = bitdepth

        # "Core" CoolChic codec. This will be reset by the warm-up function
        self.coolchic_encoder = CoolChicEncoder(self.coolchic_encoder_param)
        self.inter_coding_module = InterCodingModule(self.frame_type)

    def forward(
        self,
        reference_frames: Optional[List[Tensor]] = None,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: Optional[float] = 0.3,
        noise_parameter: Optional[float] = 1.0,
        AC_MAX_VAL: int = -1,
        flag_additional_outputs: bool = False,
    ) -> FrameEncoderOutput:
        """Perform the entire forward pass of a video frame / image.

        1. **Simulate Cool-chic decoding** to obtain both the decoded image
           :math:`\\hat{\\mathbf{x}}` as a :math:`(B, 3, H, W)` tensor
           and its associated rate :math:`\\mathrm{R}(\\hat{\\mathbf{x}})` as
           as :math:`(N)` tensor`, where :math:`N` is the number of latent
           pixels. The rate is given in bits.

        2. **Simulate the saving of the image to a file (Optional)**.
            *Only if the model has been set in test mode* e.g.
            ``self.set_to_eval()`` . Take into account that
            :math:`\\hat{\\mathbf{x}}` is a float Tensor, which is
            gonna be saved as integer values in a file.

            .. math::

                \\hat{\\mathbf{x}}_{saved} = \\mathtt{round}(\Delta_q \\
                \\hat{\\mathbf{x}}) / \\Delta_q, \\text{ with }
                \\Delta_q = 2^{bitdepth} - 1

        3. **Downscale to YUV 420 (Optional)**. *Only if the required output
           format is YUV420*. The current output is a dense Tensor. Downscale
           the last two channels to obtain a YUV420-like representation. This
           is done with a nearest neighbor downsampling.

        4. **Clamp the output** to be in :math:`[0, 1]`.

        Args:
            reference_frames: List of tensors representing the reference
                frames. Can be set to None if no reference frame is available.
                Default to None.
            quantizer_noise_type: Defaults to ``"kumaraswamy"``.
            quantizer_type: Defaults to ``"softround"``.
            soft_round_temperature: Soft round temperature.
                This is used for softround modes as well as the
                ste mode to simulate the derivative in the backward.
                Defaults to 0.3.
            noise_parameter: noise distribution parameter. Defaults to 1.0.
            AC_MAX_VAL: If different from -1, clamp the value to be in
                :math:`[-AC\\_MAX\\_VAL; AC\\_MAX\\_VAL + 1]` to write the actual bitstream.
                Defaults to -1.
            flag_additional_outputs: True to fill
                ``CoolChicEncoderOutput['additional_data']`` with many different
                quantities which can be used to analyze Cool-chic behavior.
                Defaults to False.

        Returns:
            Output of the FrameEncoder for the forward pass.
        """
        # CoolChic forward pass
        coolchic_encoder_output = self.coolchic_encoder.forward(
            quantizer_noise_type=quantizer_noise_type,
            quantizer_type=quantizer_type,
            soft_round_temperature=soft_round_temperature,
            noise_parameter=noise_parameter,
            AC_MAX_VAL=AC_MAX_VAL,
            flag_additional_outputs=flag_additional_outputs,
        )

        # Combine CoolChic output and reference frames through the inter coding modules
        inter_coding_output = self.inter_coding_module.forward(
            coolchic_output=coolchic_encoder_output,
            references=[] if reference_frames is None else reference_frames,
            flag_additional_outputs=flag_additional_outputs,
        )

        # Clamp decoded image & down sample YUV channel if needed
        if self.training:
            decoded_image = inter_coding_output.decoded_image
        else:
            max_dynamic = 2 ** (self.bitdepth) - 1
            decoded_image = (
                torch.round(inter_coding_output.decoded_image * max_dynamic)
                / max_dynamic
            )

        if self.frame_data_type == "yuv420":
            decoded_image = convert_444_to_420(decoded_image)
            decoded_image = yuv_dict_clamp(decoded_image, min_val=0.0, max_val=1.0)
        else:
            decoded_image = torch.clamp(decoded_image, 0.0, 1.0)

        additional_data = {}
        if flag_additional_outputs:
            additional_data.update(coolchic_encoder_output.get("additional_data"))
            additional_data.update(inter_coding_output.additional_data)

        results = FrameEncoderOutput(
            decoded_image=decoded_image,
            rate=coolchic_encoder_output.get("rate"),
            additional_data=additional_data,
        )

        return results

    # ------- Getter / Setter and Initializer
    def get_param(self) -> OrderedDict[str, Tensor]:
        """Return **a copy** of the weights and biases inside the module.

        Returns:
            OrderedDict[str, Tensor]: A copy of all weights & biases in the module.
        """
        param = OrderedDict({})
        param.update(
            {
                f"coolchic_encoder.{k}": v
                for k, v in self.coolchic_encoder.get_param().items()
            }
        )

        return param

    def set_param(self, param: OrderedDict[str, Tensor]):
        """Replace the current parameters of the module with param.

        Args:
            param (OrderedDict[str, Tensor]): Parameters to be set.
        """
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
        """Reinitialize in place the different parameters of a FrameEncoder."""
        self.coolchic_encoder.reinitialize_parameters()

    def set_to_train(self) -> None:
        """Set the current model to training mode, in place. This only
        affects the quantization.
        """
        self = self.train()
        self.coolchic_encoder = self.coolchic_encoder.train()
        self.inter_coding_module = self.inter_coding_module.train()

    def set_to_eval(self) -> None:
        """Set the current model to test mode, in place. This only
        affects the quantization.
        """
        self = self.eval()
        self.coolchic_encoder = self.coolchic_encoder.eval()
        self.inter_coding_module = self.inter_coding_module.eval()

    def to_device(self, device: POSSIBLE_DEVICE) -> None:
        """Push a model to a given device.

        Args:
            device: The device on which the model should run.
        """
        assert device in typing.get_args(
            POSSIBLE_DEVICE
        ), f"Unknown device {device}, should be in {typing.get_args(POSSIBLE_DEVICE)}"

        self = self.to(device)
        self.coolchic_encoder.to_device(device)

    def save(self) -> BytesIO:
        """Save the FrameEncoder into a bytes buffer and return it.

        Returns:
            Bytes representing the saved coolchic model
        """
        buffer = BytesIO()
        data_to_save = {
            "bitdepth": self.bitdepth,
            "frame_type": self.frame_type,
            "frame_data_type": self.frame_data_type,
            "coolchic_encoder_param": self.coolchic_encoder_param,
            "coolchic_encoder": self.coolchic_encoder.get_param(),
            "coolchic_nn_q_step": self.coolchic_encoder.get_network_quantization_step(),
            "coolchic_nn_expgol_cnt": self.coolchic_encoder.get_network_expgol_count(),
        }

        if self.coolchic_encoder.full_precision_param is not None:
            data_to_save["coolchic_full_precision_param"] = self.coolchic_encoder.full_precision_param

        torch.save(data_to_save, buffer)

        # for k, v in self.coolchic_encoder.get_param().items():
        #     print(f"{k:>30}: {v.abs().sum().item()}")

        return buffer

def load_frame_encoder(raw_bytes: BytesIO) -> FrameEncoder:
    """From already loaded raw bytes, load & return a CoolChicEncoder

    Args:
        raw_bytes: Already loaded raw bytes from which we'll instantiate the
            CoolChicEncoder.

    Returns:
        Frame encoder loaded by the function
    """
    # Reset the stream position to the beginning of the BytesIO object & load it
    raw_bytes.seek(0)
    loaded_data = torch.load(raw_bytes, map_location="cpu")

    # Create a frame encoder from the stored parameters
    frame_encoder = FrameEncoder(
        coolchic_encoder_param=loaded_data["coolchic_encoder_param"],
        frame_type=loaded_data["frame_type"],
        frame_data_type=loaded_data["frame_data_type"],
        bitdepth=loaded_data["bitdepth"],
    )

    # Load the different submodules (only one cool-chic for now)
    frame_encoder.coolchic_encoder.set_param(loaded_data["coolchic_encoder"])
    frame_encoder.coolchic_encoder.nn_q_step = loaded_data["coolchic_nn_q_step"]
    # Check if coolchic_nn_expgol_cnt is present in loaded data for backward
    # compatibility. Not meant to stay very long.
    if "coolchic_nn_expgol_cnt" in loaded_data:
        frame_encoder.coolchic_encoder.nn_expgol_cnt = loaded_data["coolchic_nn_expgol_cnt"]

    if "coolchic_full_precision_param" in loaded_data:
        frame_encoder.coolchic_encoder.full_precision_param = loaded_data["coolchic_full_precision_param"]

    return frame_encoder
