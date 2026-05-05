from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, List, Optional
import typing

from coolchic.bitstream.header.element import (
    HeaderElement,
    HeaderElementDescriptorCoolChic,
    HeaderElementList,
    HeaderElementSynLayer,
)

from coolchic.component.core.coolchic import CoolChicEncoderParameter
from coolchic.component.core.types import DescriptorCoolChic
from coolchic.component.frame import FrameEncoder
from coolchic.nnquant.expgolomb import POSSIBLE_EXP_GOL_COUNT
from coolchic.nnquant.quantstep import POSSIBLE_Q_STEP
from coolchic.io.types import FRAME_DATA_TYPE, POSSIBLE_BITDEPTH
import torch
from coolchic.utils.codingstructure import FRAME_TYPE, CodingStructure


@dataclass
class AbstractHeader(ABC):
    fixed_length_header: List[HeaderElement] = field(init=False, default_factory=lambda: [])

    # This can only be filled once we've read the fixed_length header e.g.,
    # the number of Element required to describe the synthesis layers can
    # only be known once the number of synthesis layers is known
    variable_length_header: List[HeaderElement] = field(init=False, default_factory=lambda: [])

    def __post__init__(self):
        self.fixed_length_header.append(HeaderElement(name="n_bytes_header", n_bits=16))

        # Count the total number of bytes in the header at the very end
        self._set_total_bytes_header()

    def _get_all_header_elements(self) -> List[HeaderElement]:
        return self.fixed_length_header + self.variable_length_header

    def _set_total_bytes_header(self) -> None:
        """Set the total number of bytes taken by the header in the "n_bytes_header"
        elements. This is achieved by iterating through all the header elements and
        summing up the number of bits they require.
        """
        n_bits = 0
        for header_element in self._get_all_header_elements():
            n_bits += header_element.n_bits

        n_bytes = math.ceil(n_bits / 8)
        self.set_value("n_bytes_header", n_bytes)

    def get_value(self, key: str) -> Optional[Any]:
        # Look into all header elements to find key == element.name
        for header_element in self._get_all_header_elements():
            if key == header_element.name:
                return header_element.get_value()

        # Return None if nothing is found
        return None

    def set_value(self, key: str, val: Any) -> None:
        # Look into all header elements to find key == element.name
        for header_element in self._get_all_header_elements():
            if key == header_element.name:
                header_element.set_value(val)
                return

        raise ValueError(f"Can not set value {val}. Key {key} can not be found in the header.")

    def read_header(self, raw_data: bytes) -> bytes:
        binary_string = "".join(format(byte, "08b") for byte in raw_data)

        for header_element in self.fixed_length_header:
            # Read the bits and set the value, also skip the read bits
            binary_string = header_element.set_bits(binary_string)

        # Create the (empty for now fields) in the variable length part of the header
        self._update_variable_length_header()
        # Read them
        for header_element in self.variable_length_header:
            # Read the bits and set the value, also skip the read bits
            binary_string = header_element.set_bits(binary_string)

        # The bytes already read are now skipped in raw_data
        raw_data = raw_data[self.get_value("n_bytes_header") :]
        return raw_data

    def to_bytes(self) -> bytes:
        binary_string = ""
        for header_element in self._get_all_header_elements():
            binary_string = f"{binary_string}{header_element.bits}"

        # Add padding bit as a suffix to the whole message so that we have
        # a number of bytes which is a multiple of 8.
        # This suffix is skipped in the read_header function.
        n_padding_bits = (8 - len(binary_string) % 8) % 8
        binary_string = binary_string + "0" * n_padding_bits

        bytes_to_write = bytearray(
            [int(binary_string[i : i + 8], base=2) for i in range(0, len(binary_string), 8)]
        )

        return bytes_to_write

    def pretty_string(self) -> str:
        msg = ""
        for header_element in self._get_all_header_elements():
            value = header_element.get_value()
            if isinstance(header_element, HeaderElementDescriptorCoolChic):
                val_str = value.pretty_string()
            else:
                val_str = f"{value}"

            msg += f"{header_element.name:<30}{val_str:<40}\n"
        return msg

    @abstractmethod
    def _update_variable_length_header(self) -> None:
        pass

    @abstractmethod
    def set_header(self) -> None:
        pass


@dataclass
class VideoHeader(AbstractHeader):
    def __post_init__(self):
        self.fixed_length_header += [
            # ---- Image information
            HeaderElement(name="n_frames", n_bits=12),
            HeaderElement(name="n_intras", n_bits=12),
            HeaderElement(name="n_p_frames", n_bits=12),
        ]

        # At the end since it computes the number of bytes in the Header
        super().__post__init__()

    def _update_variable_length_header(self) -> None:
        self.variable_length_header += [
            HeaderElementList(
                name="intra_pos", n_bits_per_val=12, n_val=self.get_value("n_intras")
            ),
            HeaderElementList(name="p_pos", n_bits_per_val=12, n_val=self.get_value("n_p_frames")),
        ]

    def set_header(self, coding_structure: CodingStructure) -> None:
        self.set_value("n_frames", coding_structure.n_frames)
        self.set_value("n_intras", len(coding_structure.intra_pos))
        self.set_value("n_p_frames", len(coding_structure.p_pos))
        self._update_variable_length_header()

        self.set_value("intra_pos", coding_structure.intra_pos)
        self.set_value("p_pos", coding_structure.p_pos)

        # ---- Count the overall number of bytes and write it into the corresponding field
        self._set_total_bytes_header()

    def get_coding_structure(self) -> CodingStructure:

        return CodingStructure(
            n_frames=self.get_value("n_frames"),
            intra_pos=self.get_value("intra_pos"),
            p_pos=self.get_value("p_pos"),
        )


@dataclass
class FrameHeader(AbstractHeader):
    def __post_init__(self):
        self.fixed_length_header += [
            # ---- Image information
            HeaderElement(name="display_index", n_bits=12),
            HeaderElement(
                name="frame_type",
                n_bits=2,
                idx_to_value=True,
                possible_values=typing.get_args(FRAME_TYPE),
            ),
            HeaderElement(
                name="frame_data_type",
                n_bits=2,
                idx_to_value=True,
                possible_values=typing.get_args(FRAME_DATA_TYPE),
            ),
            HeaderElement(
                name="bitdepth",
                n_bits=4,
                idx_to_value=True,
                possible_values=typing.get_args(POSSIBLE_BITDEPTH),
            ),
        ]

        # At the end since it computes the number of bytes in the Header
        super().__post__init__()

    def _update_variable_length_header(self) -> None:

        frame_type = self.get_value("frame_type")
        if frame_type == "B":
            n_refs = 2
        elif frame_type == "P":
            n_refs = 1
        else:
            n_refs = 0

        self.variable_length_header += [
            HeaderElementList(name="index_references", n_bits_per_val=12, n_val=n_refs),
            # There is a *2 because there is a flow_x and flow_y
            HeaderElementList(name="global_flow", n_bits_per_val=14, n_val=n_refs * 2, signed=True),
        ]

        if frame_type in ["P", "B"]:
            self.variable_length_header += [
                HeaderElement(name="warp_filter_size", n_bits=4),
            ]

    def set_header(self, frame_encoder: FrameEncoder) -> None:
        self.set_value("display_index", frame_encoder.frame_display_index)
        self.set_value("frame_type", frame_encoder.frame_type)
        self.set_value("frame_data_type", frame_encoder.frame_data_type)
        self.set_value("bitdepth", frame_encoder.bitdepth)

        self._update_variable_length_header()

        if frame_encoder.frame_type in ["P", "B"]:
            self.set_value("index_references", frame_encoder.index_references)

            global_flow = frame_encoder.global_flow_1.view(-1)
            if frame_encoder.frame_type == "B":
                global_flow = torch.cat([global_flow, frame_encoder.global_flow_2.view(-1)], dim=0)
            global_flow = global_flow.to(torch.int).tolist()
            self.set_value("global_flow", global_flow)
            self.set_value("warp_filter_size", frame_encoder.warp_parameter.filter_size)

        # ---- Count the overall number of bytes and write it into the corresponding field
        self._set_total_bytes_header()


@dataclass
class CoolChicHeader(AbstractHeader):
    def __post_init__(self):
        self.fixed_length_header += [
            # ---- Synthesis
            HeaderElement(name="linear_stabiliser_synth", n_bits=1),
            HeaderElement(name="n_layer_synthesis", n_bits=3),
            # ---- Upsampling
            HeaderElement(name="ups_k_size", n_bits=4),
            HeaderElement(name="ups_preconcat_k_size", n_bits=4),
            # ---- Entropy model
            HeaderElement(name="output_feature_ifce", n_bits=5),
            HeaderElement(name="spatial_context_arm", n_bits=6),
            HeaderElement(name="linear_stabiliser_arm", n_bits=1),
            HeaderElement(name="n_hidden_layers_arm", n_bits=3),
            # ---- Latent grid and hyperlatent grids
            HeaderElementList(name="img_size", n_bits_per_val=14, n_val=2),
            HeaderElementList(name="latent_resolution", n_bits_per_val=4, n_val=2),
            HeaderElement(name="n_latent_grids", n_bits=5),
            HeaderElement(name="flag_hyperlatent", n_bits=1),
            HeaderElement(name="flag_common_randomness", n_bits=1),
            # ---- Others
            HeaderElement(
                name="final_upsampling_type",
                n_bits=2,
                idx_to_value=True,
                possible_values=["nearest", "bilinear", "bicubic"],
            ),
            # ---- Neural network quantization parameters
            HeaderElementDescriptorCoolChic(
                name="nn_q_step",
                n_bits_per_val=5,
                idx_to_value=True,
                possible_values=POSSIBLE_Q_STEP,
            ),
            HeaderElementDescriptorCoolChic(
                name="nn_expgol_cnt",
                n_bits_per_val=4,
                idx_to_value=True,
                possible_values=POSSIBLE_EXP_GOL_COUNT,
            ),
            HeaderElement(name="nn_n_bytes", n_bits=14),
            HeaderElement(name="nn_n_bit_pad", n_bits=3),
            HeaderElement(name="n_bytes_latent", n_bits=28),
        ]

        # At the end since it computes the number of bytes in the Header
        super().__post__init__()

    def _update_variable_length_header(self) -> None:

        if self.get_value("output_feature_ifce") > 0:
            self.variable_length_header += [
                HeaderElementList(name="ifce_resolution", n_bits_per_val=4, n_val=2)
            ]

        if self.get_value("flag_hyperlatent"):
            self.variable_length_header += [
                HeaderElementList(name="hyperlatent_resolution", n_bits_per_val=4, n_val=2)
            ]

        # Add synthesis layer description
        self.variable_length_header += [
            HeaderElementSynLayer(name=f"syn_layer_{idx_layer}")
            for idx_layer in range(self.get_value("n_layer_synthesis"))
        ]

    def set_header(
        self,
        cc_enc_param: CoolChicEncoderParameter,
        nn_n_bytes: int,
        nn_n_bit_pad: int,
        nn_q_step: DescriptorCoolChic,
        nn_expgol_cnt: DescriptorCoolChic,
        n_bytes_latent: int,
    ) -> None:
        # ----- Save the value for the fixed-length part of the header
        self.set_value("linear_stabiliser_synth", cc_enc_param.linear_stabiliser_synth)
        self.set_value("n_layer_synthesis", len(cc_enc_param.layers_synthesis))
        self.set_value("ups_k_size", cc_enc_param.ups_k_size)
        self.set_value("ups_preconcat_k_size", cc_enc_param.ups_preconcat_k_size)
        self.set_value("output_feature_ifce", cc_enc_param.output_feature_ifce)
        self.set_value("spatial_context_arm", cc_enc_param.spatial_context_arm)
        self.set_value("linear_stabiliser_arm", cc_enc_param.linear_stabiliser_arm)
        self.set_value("n_hidden_layers_arm", cc_enc_param.n_hidden_layers_arm)
        self.set_value("img_size", cc_enc_param.img_size)
        self.set_value("latent_resolution", cc_enc_param.latent_resolution)
        self.set_value("n_latent_grids", cc_enc_param.n_latent_grids)
        self.set_value("flag_hyperlatent", cc_enc_param.flag_hyperlatent)
        self.set_value("flag_common_randomness", cc_enc_param.flag_common_randomness)
        self.set_value("final_upsampling_type", cc_enc_param.final_upsampling_type)

        self.set_value("nn_q_step", nn_q_step)
        self.set_value("nn_expgol_cnt", nn_expgol_cnt)
        self.set_value("nn_n_bytes", nn_n_bytes)
        self.set_value("nn_n_bit_pad", nn_n_bit_pad)
        self.set_value("n_bytes_latent", n_bytes_latent)

        # ----- Save the value for the variable-length part of the header
        self._update_variable_length_header()
        for idx_layer in range(self.get_value("n_layer_synthesis")):
            self.set_value(f"syn_layer_{idx_layer}", cc_enc_param.layers_synthesis[idx_layer])

        if cc_enc_param.flag_ifce:
            self.set_value("ifce_resolution", cc_enc_param.ifce_resolution)

        if cc_enc_param.flag_hyperlatent:
            self.set_value("hyperlatent_resolution", cc_enc_param.hyperlatent_resolution)

        # ---- Count the overall number of bytes and write it into the corresponding field
        self._set_total_bytes_header()

    def get_coolchic_parameter(self) -> CoolChicEncoderParameter:
        """Use the information in the header to reconstruct a CoolChicEncoderParameter
        object, ready to instantiate a CoolChic module.
        """
        layer_synthesis = [
            self.get_value(f"syn_layer_{i}") for i in range(self.get_value("n_layer_synthesis"))
        ]
        param = CoolChicEncoderParameter(
            layers_synthesis=layer_synthesis,
            linear_stabiliser_synth=self.get_value("linear_stabiliser_synth"),
            ups_k_size=self.get_value("ups_k_size"),
            ups_preconcat_k_size=self.get_value("ups_preconcat_k_size"),
            ifce_resolution=self.get_value("ifce_resolution"),
            output_feature_ifce=self.get_value("output_feature_ifce"),
            spatial_context_arm=self.get_value("spatial_context_arm"),
            linear_stabiliser_arm=self.get_value("linear_stabiliser_arm"),
            n_hidden_layers_arm=self.get_value("n_hidden_layers_arm"),
            latent_resolution=self.get_value("latent_resolution"),
            hyperlatent_resolution=self.get_value("hyperlatent_resolution"),
            flag_common_randomness=self.get_value("flag_common_randomness"),
            img_size=self.get_value("img_size"),
            final_upsampling_type=self.get_value("final_upsampling_type"),
        )
        return param
