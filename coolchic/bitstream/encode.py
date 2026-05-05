# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import subprocess

from coolchic.bitstream.header.header import CoolChicHeader, FrameHeader, VideoHeader
import torch

from coolchic.bitstream.component.coolchic import encode_decode_coolchic
from coolchic.bitstream.component.constants import AC_MAX_VAL
from coolchic.bitstream.neuralnet.neuralnet import encode_network
from coolchic.component.frame import FrameEncoder
from coolchic.utils.codingstructure import CodingStructure


@torch.no_grad()
@torch.inference_mode()
def encode_frame(
    frame_encoder: FrameEncoder,
    bitstream_path: str,
    coding_structure: CodingStructure,
    verbosity: int = 0,
) -> bytes:
    """Convert a model to a bitstream located at <bitstream_path>.

    Return the bytes corresponding to the frame.

    Args:
        model (CoolChicEncoder): A trained and quantized model
        bitstream_path (str): Where to save the bitstream
    """

    torch.use_deterministic_algorithms(True)
    frame_encoder.set_to_eval()
    frame_encoder.to_device("cpu")

    bytes_to_write = b""

    # Delete existing bytes only for the very first video frame
    if frame_encoder.frame_display_index == 0:
        subprocess.call(f"rm -f {bitstream_path}", shell=True)
        video_header = VideoHeader()
        video_header.set_header(coding_structure)
        bytes_to_write += video_header.to_bytes()
        if verbosity:
            print(video_header.pretty_string())

    frame_header = FrameHeader()
    frame_header.set_header(frame_encoder)
    if verbosity:
        print(frame_header.pretty_string())

    bytes_to_write += frame_header.to_bytes()

    all_cc_name = ["residue"]
    if frame_header.get_value("frame_type") in ["P", "B"]:
        all_cc_name.append("motion")

    for cc_name in all_cc_name:
        cc_enc = frame_encoder.coolchic_enc[cc_name]
        bytes_nn, descriptor_nn = encode_network(cc_enc)
        quantized_latent = cc_enc.get_quantize_latent(
            quantizer_noise_type="none", quantizer_type="hardround", AC_MAX_VAL=AC_MAX_VAL
        )
        header = CoolChicHeader()
        header.set_header(
            cc_enc.param,
            descriptor_nn.get("nn_n_bytes"),
            descriptor_nn.get("nn_n_bit_pad"),
            descriptor_nn.get("nn_q_step"),
            descriptor_nn.get("nn_expgol_cnt"),
            # We don't now the number of bytes in the latent grids yet, it'll be filled by
            # the encode_decode_coolchic function
            n_bytes_latent=0,
        )

        _, bytes_coolchic = encode_decode_coolchic(
            header,
            bytes_nn,
            mode="encode",
            dec_bytes_latent=None,
            enc_quantized_latent=quantized_latent,
        )
        bytes_to_write += bytes_coolchic

    with open(bitstream_path, "ab") as f_out:
        f_out.write(bytes_to_write)

    return bytes_to_write
