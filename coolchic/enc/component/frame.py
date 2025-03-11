# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


"""A frame encoder is composed of one or two CoolChicEncoder."""

import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

from enc.component.types import DescriptorCoolChic, NAME_COOLCHIC_ENC
from enc.utils.termprint import center_str
import torch
import torch.nn.functional as F
from enc.component.coolchic import (
    CoolChicEncoder,
    CoolChicEncoderParameter,
)
from enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
)
from enc.component.intercoding.warp import warp_fn
from enc.io.types import FRAME_DATA_TYPE, POSSIBLE_BITDEPTH
from enc.io.format.yuv import DictTensorYUV, convert_444_to_420, yuv_dict_clamp
from enc.utils.codingstructure import FRAME_TYPE
from enc.training.manager import FrameEncoderManager
from enc.utils.device import POSSIBLE_DEVICE
from torch import Tensor, nn


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

    # Rate associated to each cool-chic encoder
    rate: Dict[NAME_COOLCHIC_ENC, Tensor]

    # Any other data required to compute some logs, stored inside a dictionary
    additional_data: Dict[str, Any] = field(default_factory=lambda: {})


class FrameEncoder(nn.Module):
    """A ``FrameEncoder`` is the object containing everything
    required to encode a video frame or an image. It is composed of
    one or more ``CoolChicEncoder``.
    """

    def __init__(
        self,
        coolchic_enc_param: Dict[NAME_COOLCHIC_ENC, CoolChicEncoderParameter],
        frame_type: FRAME_TYPE = "I",
        frame_data_type: FRAME_DATA_TYPE = "rgb",
        bitdepth: POSSIBLE_BITDEPTH = 8,
        index_references: List[int] = [],
        frame_display_index: int = 0,
    ):
        """
        Args:
            coolchic_enc_param: Parameters for the underlying CoolChicEncoders
            frame_type: More info in
                :doc:`coding_structure.py <../utils/codingstructure>`.
                Defaults to "I".
            frame_data_type: More info in
                :doc:`coding_structure.py <../utils/codingstructure>`.
                Defaults to "rgb"
            bitdepth: More info in
                :doc:`coding_structure.py <../utils/codingstructure>`.
                Defaults to 8.
            index_references: List of the display index of the references.
                Defaults to []
            frame_display_index: display index of the frame being encoded.
        """
        super().__init__()

        # ----- Copy the parameters
        self.coolchic_enc_param = coolchic_enc_param
        self.frame_type = frame_type
        self.frame_data_type = frame_data_type
        self.bitdepth = bitdepth
        self.index_references = index_references
        self.frame_display_index = frame_display_index

        # Check we've passed the expected number of frames.
        all_expected_n_ref = {"I": 0, "P": 1, "B": 2}
        for frame_type, expected_n_ref in all_expected_n_ref.items():
            if self.frame_type == frame_type:
                assert len(self.index_references) == expected_n_ref, (
                    f"{frame_type} frame must have {expected_n_ref} references. "
                    f"Found {len(self.index_references)}: {self.index_references}."
                )

        # "Core" CoolChic codec. This will be reset by the warm-up function
        self.coolchic_enc: Dict[NAME_COOLCHIC_ENC, CoolChicEncoder] = nn.ModuleDict()
        for name, cc_enc_param in self.coolchic_enc_param.items():
            self.coolchic_enc[name] = CoolChicEncoder(cc_enc_param)

        # Global motion. Only here for saving purposes. Not used in the forward
        # We shift the references instead!

        # Global motion --> Shift the entire ref by a constant motion prior to
        # using the optical flow recovered from the motion cool-chic.
        # register_buffer for automatic device management. We set persistent to false
        # to simply use the "automatically move to device" function, without
        # considering global_flow_1 as a parameters (i.e. returned
        # by self.parameters())

        self.register_buffer("global_flow_1", torch.zeros(1, 2, 1, 1), persistent=False)
        self.register_buffer("global_flow_2", torch.zeros(1, 2, 1, 1), persistent=False)

        # self.global_flow_1 = nn.Parameter(torch.zeros(1, 2, 1, 1), requires_grad=True)
        # self.global_flow_2 = nn.Parameter(torch.zeros(1, 2, 1, 1), requires_grad=True)


    def forward(
        self,
        reference_frames: Optional[List[Tensor]] = None,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: Optional[Tensor] = torch.tensor(0.3),
        noise_parameter: Optional[Tensor] = torch.tensor(1.0),
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

        # Common parameters for all cool-chic encoders
        cc_forward_param = {
            "quantizer_noise_type": quantizer_noise_type,
            "quantizer_type": quantizer_type,
            "soft_round_temperature": soft_round_temperature,
            "noise_parameter": noise_parameter,
            "AC_MAX_VAL": AC_MAX_VAL,
            "flag_additional_outputs": flag_additional_outputs,
        }

        cc_enc_out = {
            cc_name: cc_enc(**cc_forward_param)
            for cc_name, cc_enc in self.coolchic_enc.items()
        }

        # Get the rate of each cool-chic encoder
        rate = {
            cc_name: cc_enc_out_i.get("rate")
            for cc_name, cc_enc_out_i in cc_enc_out.items()
        }

        if self.frame_type == "I":
            decoded_image = cc_enc_out["residue"].get("raw_out")

        elif self.frame_type in ["P", "B"]:
            residue = cc_enc_out["residue"].get("raw_out")[:, :3, :, :]
            alpha = torch.clamp(
                cc_enc_out["residue"].get("raw_out")[:, 3:4, :, :] + 0.5, 0.0, 1.0
            )
            flow_1 = cc_enc_out["motion"].get("raw_out")[:, 0:2, :, :]

            # Apply each global flow on each reference.
            # Upsample the global flow beforehand to obtain a constant [1, 2, H, W] optical flow.
            shifted_ref = []
            for ref_i, global_flow_i in zip(reference_frames, [self.global_flow_1, self.global_flow_2]):
                ups_global_flow_i = F.interpolate(global_flow_i, size=ref_i.size()[-2:], mode="nearest")
                shifted_ref.append(warp_fn(ref_i, ups_global_flow_i))

            if self.frame_type == "P":
                pred = warp_fn(shifted_ref[0], flow_1)

            elif self.frame_type == "B":
                flow_2 = cc_enc_out["motion"].get("raw_out")[:, 2:4, :, :]
                beta = torch.clamp(
                    cc_enc_out["motion"].get("raw_out")[:, 4:5, :, :] + 0.5, 0.0, 1.0
                )
                pred = beta * warp_fn(shifted_ref[0], flow_1) \
                       + (1 - beta) * warp_fn( shifted_ref[1], flow_2)

            decoded_image = alpha * pred + residue

        # Clamp decoded image & down sample YUV channel if needed
        if not self.training:
            max_dynamic = 2 ** (self.bitdepth) - 1
            decoded_image = torch.round(decoded_image * max_dynamic) / max_dynamic

        if self.frame_data_type == "yuv420":
            decoded_image = convert_444_to_420(decoded_image)
            decoded_image = yuv_dict_clamp(decoded_image, min_val=0.0, max_val=1.0)
        elif self.frame_data_type != "flow":
            decoded_image = torch.clamp(decoded_image, 0.0, 1.0)

        additional_data = {}
        if flag_additional_outputs:
            # Browse all the cool-chic output to get their additional data
            for cc_name, cc_enc_out_i in cc_enc_out.items():
                additional_data.update(
                    {
                        # Append the cc_name (e.g. residue) in front of the key
                        f"{cc_name}{k}": v
                        for k, v in cc_enc_out_i.get("additional_data").items()
                    }
                )

            # Also add the residue, flow, pred and beta
            if self.frame_type in ["P", "B"]:
                additional_data["residue"] = residue
                additional_data["alpha"] = alpha
                additional_data["flow_1"] = flow_1
                additional_data["pred"] = pred

            if self.frame_type == "B":
                additional_data["flow_2"] = flow_2
                additional_data["beta"] = beta

        results = FrameEncoderOutput(
            decoded_image=decoded_image,
            rate=rate,
            additional_data=additional_data,
        )

        return results

    # ------- Getter / Setter and Initializer
    def get_param(self) -> OrderedDict[NAME_COOLCHIC_ENC, Tensor]:
        """Return **a copy** of the weights and biases inside the module.

        Returns:
            OrderedDict[NAME_COOLCHIC_ENC, Tensor]: A copy of all weights & biases in the module.
        """
        param = OrderedDict({})

        for cc_name, cc_enc in self.coolchic_enc.items():
            param.update(
                {
                    f"coolchic_enc.{cc_name}.{k}": v
                    for k, v in cc_enc.get_param().items()
                }
            )

        return param

    def set_param(self, param: OrderedDict[NAME_COOLCHIC_ENC, Tensor]):
        """Replace the current parameters of the module with param.

        Args:
            param (OrderedDict[NAME_COOLCHIC_ENC, Tensor]): Parameters to be set.
        """
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
        """Reinitialize in place the different parameters of a FrameEncoder."""
        for _, cc_enc in self.coolchic_enc.items():
            print("CHECK THAT I DO RE-INIT THE PARAM!")
            cc_enc.reinitialize_parameters()

    def _store_full_precision_param(self) -> None:
        """For all the coolchic_enc,  store their current parameters inside
        self.full_precision_param.

        This function checks that there is no self.nn_q_step and
        self.nn_expgol_cnt already saved. This would mean that we no longer
        have full precision parameters but quantized ones.
        """
        for _, cc_enc in self.coolchic_enc.items():
            cc_enc._store_full_precision_param()

    def set_to_train(self) -> None:
        """Set the current model to training mode, in place. This only
        affects the quantization.
        """
        self = self.train()
        for _, cc_enc in self.coolchic_enc.items():
            cc_enc.train()

    def set_global_flow(self, global_flow_1: Tensor, global_flow_2: Tensor) -> None:
        """Set the value of the global flows.

        The global flows are 2-element tensors. The first one is the horizontal
        displacement and the second one the vertical displacement.

        Args:
            global_flow_1 (Tensor): Value of global flow for reference 1. Must have 2 elements.
            global_flow_2 (Tensor): Value of global flow for reference 2. Must have 2 elements.
        """

        assert global_flow_1.numel() == 2, (
            f"global_flow_1 must have 2 parameters. Found {global_flow_1.numel()} "
            " parameters."
        )

        assert global_flow_2.numel() == 2, (
            f"global_flow_2 must have 2 parameters. Found {global_flow_2.numel()} "
            " parameters."
        )

        self.global_flow_1 = global_flow_1.view(self.global_flow_1.size())
        self.global_flow_2 = global_flow_2.view(self.global_flow_2.size())

    def get_network_rate(self) -> Tuple[Dict[NAME_COOLCHIC_ENC, DescriptorCoolChic], int]:
        """Return the rate (in bits) associated to the parameters
        (weights and biases) of the different modules

        Returns:
            Tuple[Dict[NAME_COOLCHIC_ENC, DescriptorCoolChic], int]: The rate (in bits)
            associated with the weights and biases of each module of each
            cool-chic decoder. Also return the overall rate in bits.
        """

        detailed_rate_bit = {}
        total_rate_bit = 0.0

        for cc_name, cc_enc in self.coolchic_enc.items():
            detailed_rate_bit[cc_name], sum_rate = cc_enc.get_network_rate()
            total_rate_bit += sum_rate

        return detailed_rate_bit, total_rate_bit

    def get_network_quantization_step(
        self,
    ) -> Dict[NAME_COOLCHIC_ENC, DescriptorCoolChic]:
        """Return the quantization step associated to the parameters (weights
        and biases) of the different modules of each cool-chic decoder. Those
        quantization can be ``None`` if the model has not yet been quantized.

        E.g. {"residue": {"arm": 4, "upsampling": 12, "synthesis": 1}}

        Returns:
            Dict[NAME_COOLCHIC_ENC, DescriptorCoolChic]: The quantization step
            associated with the weights and biases of each module of each
            cool-chic decoder.
        """

        q_step = {}
        for cc_name, cc_enc in self.coolchic_enc.items():
            q_step[cc_name] = cc_enc.get_network_quantization_step()

        return q_step

    def get_network_expgol_count(self) -> Dict[NAME_COOLCHIC_ENC, DescriptorCoolChic]:
        """Return the Exp-Golomb count parameter associated to the parameters
        (weights and biases) of the different modules of each cool-chic decoder.
        Those exp-golomb param can be ``None`` if the model has not yet
        been quantized.

        E.g. {"residue": {"arm": 4, "upsampling": 12, "synthesis": 1}}

        Returns:
            Dict[NAME_COOLCHIC_ENC, DescriptorCoolChic]: The exp-golomb count
            parameter associated with the weights and biases of each module of
            each cool-chic decoder.
        """

        expgol_cnt = {}
        for cc_name, cc_enc in self.coolchic_enc.items():
            expgol_cnt[cc_name] = cc_enc.get_network_expgol_count()

        return expgol_cnt

    def get_total_mac_per_pixel(self) -> float:
        """Count the number of Multiplication-Accumulation (MAC) per decoded pixel
        for this model.

        Returns:
            float: number of floating point operations per decoded pixel.
        """

        mac_per_pixel = 0
        for cc_name, cc_enc in self.coolchic_enc.items():
            mac_per_pixel += cc_enc.get_total_mac_per_pixel()

        return mac_per_pixel

    def set_to_eval(self) -> None:
        """Set the current model to test mode, in place. This only
        affects the quantization.
        """
        self = self.eval()
        for _, cc_enc in self.coolchic_enc.items():
            cc_enc.eval()

    def to_device(self, device: POSSIBLE_DEVICE) -> None:
        """Push a model to a given device.

        Args:
            device: The device on which the model should run.
        """
        assert device in typing.get_args(
            POSSIBLE_DEVICE
        ), f"Unknown device {device}, should be in {typing.get_args(POSSIBLE_DEVICE)}"

        self = self.to(device)
        for _, cc_enc in self.coolchic_enc.items():
            cc_enc.to_device(device)

    def save(
        self,
        path_file: str,
        frame_encoder_manager: Optional[FrameEncoderManager] = None,
    ) -> None:
        """Save the FrameEncoder into a bytes buffer and return it.
            Optionally save a frame_encoder_manager alongside the current frame
            encoder to keep track of the training time, record loss etc.

        Args:
            path_file: Where to save the FrameEncoder
            frame_encoder_manager: Contains (among other things) the rate
                constraint :math:`\\lambda` and description of the
                warm-up preset. It is also used to track the total encoding time
                and encoding iterations.

            Returns:
                Bytes representing the saved coolchic model
        """
        data_to_save = {
            "bitdepth": self.bitdepth,
            "frame_type": self.frame_type,
            "frame_data_type": self.frame_data_type,
            "index_references": self.index_references,
            "frame_display_index": self.frame_display_index,
            # Name of the different cool-chic encoder
            "keys_cc_enc": list(self.coolchic_enc.keys()),
            "global_flow_1": self.global_flow_1,
            "global_flow_2": self.global_flow_2,
        }

        for cc_name, cc_enc in self.coolchic_enc.items():
            data_to_save[f"{cc_name}"] = cc_enc.get_param()
            data_to_save[f"{cc_name}_nn_q_step"] = (
                cc_enc.get_network_quantization_step()
            )
            data_to_save[f"{cc_name}_nn_expgol_cnt"] = cc_enc.get_network_expgol_count()
            data_to_save[f"{cc_name}_param"] = self.coolchic_enc_param[cc_name]

            if cc_enc.full_precision_param is not None:
                data_to_save[f"{cc_name}_full_precision_param"] = (
                    cc_enc.full_precision_param
                )

        if frame_encoder_manager is not None:
            data_to_save["frame_encoder_manager"] = frame_encoder_manager

        torch.save(data_to_save, path_file)

    def pretty_string(self, print_detailed_archi: bool = False) -> str:
        """Get a pretty string representing the architectures of
        the different ``CoolChicEncoder`` composing the current ``FrameEncoder``.

        Args:
            print_detailed_archi: True to print the detailed decoder architecture

        Returns:
            str: a pretty string ready to be printed out
        """

        s = ""

        for name, cc_enc in self.coolchic_enc.items():
            total_mac_per_pix = cc_enc.get_total_mac_per_pixel()
            title = (
                "\n\n"
                f"{name} decoder: {total_mac_per_pix:5.0f} MAC / pixel"
                "\n"
                f"{'-' * len(name)}---------------------------"
                "\n"
            )
            s += title
            s += cc_enc.pretty_string(print_detailed_archi=print_detailed_archi) + "\n"
        return s

    def pretty_string_param(self) -> str:
        """Get a pretty string representing the parameters of
        the different ``CoolChicEncoderParameters`` parameterising the current
        ``FrameEncoder``
        """

        s = ""

        for name, cc_enc_param in self.coolchic_enc_param.items():
            title = (
                "\n\n"
                + center_str(f"{name} parameters")
                + "\n"
                + center_str(f"{'-' * len(name)})-----------")
                + "\n\n"
            )
            s += title
            s += cc_enc_param.pretty_string() + "\n"
        return s


def load_frame_encoder(
    path_file: str,
) -> Tuple[FrameEncoder, Optional[FrameEncoderManager]]:
    """From already loaded raw bytes, load & return a CoolChicEncoder

    Args:
        path_file: Path of the FrameEncoder to be loaded

    Returns:
        Tuple with a FrameEncoder loaded by the function and an optional
        FrameEncoderManager
    """
    loaded_data = torch.load(path_file, map_location="cpu", weights_only=False)

    # Something like ["residue", "motion"]
    list_cc_name = loaded_data["keys_cc_enc"]

    # Load first the CoolChicEncoderParameter of all the Cool-chic encoders
    # for the frame
    coolchic_enc_param = {}
    for cc_name in list_cc_name:
        coolchic_enc_param[cc_name] = loaded_data[f"{cc_name}_param"]

    # Create a, empty frame encoder from the stored parameters
    frame_encoder = FrameEncoder(
        coolchic_enc_param=coolchic_enc_param,
        frame_type=loaded_data["frame_type"],
        frame_data_type=loaded_data["frame_data_type"],
        bitdepth=loaded_data["bitdepth"],
        index_references=loaded_data["index_references"],
        frame_display_index=loaded_data["frame_display_index"],
    )

    # Load the parameters
    for cc_name in list_cc_name:
        frame_encoder.coolchic_enc[cc_name].set_param(loaded_data[cc_name])
        frame_encoder.coolchic_enc[cc_name].nn_q_step = loaded_data[
            f"{cc_name}_nn_q_step"
        ]
        frame_encoder.coolchic_enc[cc_name].nn_expgol_cnt = loaded_data[
            f"{cc_name}_nn_expgol_cnt"
        ]

        if f"{cc_name}_full_precision_param" in loaded_data:
            frame_encoder.coolchic_enc[cc_name].full_precision_parameter = loaded_data[
                f"{cc_name}_full_precision_param"
            ]

    frame_encoder_manager = loaded_data["frame_encoder_manager"]

    if "global_flow_1" in loaded_data:
        frame_encoder.global_flow_1 = loaded_data["global_flow_1"]

    if "global_flow_2" in loaded_data:
        frame_encoder.global_flow_2 = loaded_data["global_flow_2"]

    return frame_encoder, frame_encoder_manager
