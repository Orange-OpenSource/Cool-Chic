# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import time
from typing import Dict, List, Optional, Tuple

from coolchic.bitstream.header.header import CoolChicHeader, FrameHeader, VideoHeader
import torch

from coolchic.bitstream.component.coolchic import encode_decode_coolchic
from coolchic.component.intercoding.globalmotion import apply_global_translation
from coolchic.component.intercoding.warp import Warper, WarpParameter
from coolchic.io.format.yuv import convert_420_to_444, convert_444_to_420, yuv_dict_clamp
from coolchic.io.framedata import FrameData
from coolchic.io.io import save_frame_data_to_file


@torch.no_grad()
@torch.inference_mode()
def decode_video(
    bitstream_path: str,
    decoded_path: Optional[str] = None,
    max_decoding_order: int = -1,
    verbosity: int = 0,
) -> Dict[str, FrameData]:
    """
    Decode an image or video from a bitstream and optionally save the results.
    Also return a dictionary of decoded frame indexed by display order.

    Args:
        bitstream_path (str): Path of the bitstream.
        decoded_path (Optional[str]): Path where the compressed video is saved.
            If None, do not save the frames. Just run the decoder and
            get the decoded frames in the dictionary returned by this function.
        max_decoding_order (int): Set to N to decode only the first (N + 1) frames.
            Set to -1 to decode all frames. Default to -1.
        verbosity (int, optional): Verbosity level. Defaults to 0.

    Returns:
        Dict[str, FrameData]: {
            "0": frame data for the frame with display index 0,
            "32" frame data for the frame with display index 32
        }
    """

    with open(bitstream_path, "rb") as f_in:
        bitstream_bytes = f_in.read()

    video_header = VideoHeader()
    bitstream_bytes = video_header.read_header(bitstream_bytes)
    coding_structure = video_header.get_coding_structure()

    if verbosity:
        print(video_header.pretty_string())
        print(coding_structure.pretty_structure_diagram())

    if max_decoding_order == -1:
        max_decoding_order = coding_structure.get_max_coding_order()

    # +1 because we want to decode from frame 0 to N in coding order
    for coding_idx in range(max_decoding_order + 1):
        start_time = time.time()
        frame = coding_structure.get_frame_from_coding_order(coding_idx)

        refs_data = [
            coding_structure.get_frame_from_display_order(idx_ref).data
            for idx_ref in frame.index_references
        ]
        frame_data, bitstream_bytes = decode_frame(
            bitstream_bytes, reference_frames=refs_data, verbosity=verbosity
        )
        frame.set_frame_data(frame_data)
        print(
            f"Decoding frame {frame.display_order:<4} time = {time.time() - start_time:6.2f} seconds."
        )
        start_time = time.time()

    all_frames = {}
    for display_idx in range(coding_structure.get_max_display_order() + 1):
        frame = coding_structure.get_frame_from_display_order(display_idx)
        all_frames[str(display_idx)] = frame.data
        if decoded_path is not None:
            save_frame_data_to_file(frame.data, decoded_path, append=display_idx != 0)

    return all_frames


# from line_profiler import profile
# @profile
def decode_frame(
    bitstream_bytes: bytes, reference_frames: List[FrameData], verbosity: int = 0
) -> Tuple[FrameData, bytes]:
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

    # 1. ----- Read header
    frame_header = FrameHeader()
    # Read and skip the bytes
    bitstream_bytes = frame_header.read_header(bitstream_bytes)

    if verbosity:
        print(frame_header.pretty_string())

    frame_type = frame_header.get_value("frame_type")
    bitdepth = frame_header.get_value("bitdepth")
    frame_data_type = frame_header.get_value("frame_data_type")

    all_cc_name = ["residue"]
    if frame_type in ["P", "B"]:
        all_cc_name.append("motion")

    # 2. ----- Decode all cool-chics
    cc_out = {}
    for cc_name in all_cc_name:
        # Read and skip the bytes
        cc_header = CoolChicHeader()
        bitstream_bytes = cc_header.read_header(bitstream_bytes)

        n_bytes_nn = cc_header.get_value("nn_n_bytes")
        bytes_nn = bitstream_bytes[:n_bytes_nn]
        bitstream_bytes = bitstream_bytes[n_bytes_nn:]

        n_bytes_latent = cc_header.get_value("n_bytes_latent")
        bytes_latent = bitstream_bytes[:n_bytes_latent]
        bitstream_bytes = bitstream_bytes[n_bytes_latent:]

        raw_out, _ = encode_decode_coolchic(
            cc_header,
            bytes_nn,
            mode="decode",
            dec_bytes_latent=bytes_latent,
            enc_quantized_latent=None,
            verbosity=verbosity,
        )
        cc_out[cc_name] = raw_out

    # 3. ----- Reconstruct the frame by combining the different cool-chic outputs
    if frame_type in ["I"]:
        decoded_image = cc_out["residue"]
    else:
        if frame_data_type == "yuv420":
            raw_references = [convert_420_to_444(ref_i.data) for ref_i in reference_frames]
        else:
            raw_references = [ref_i.data for ref_i in reference_frames]

        warper = Warper(
            param=WarpParameter(filter_size=frame_header.get_value("warp_filter_size")),
            img_size=raw_references[0].size()[-2:],
        )
        global_flows = torch.tensor(frame_header.get_value("global_flow"), dtype=torch.float).view(
            1, -1, 1, 1
        )
        global_flows = torch.split(global_flows, 2, dim=1)
        shifted_ref = apply_global_translation(raw_references, global_flows)

        residue = cc_out["residue"][:, :3, :, :]
        alpha = torch.clamp(cc_out["residue"][:, 3:4, :, :] + 0.5, 0.0, 1.0)
        flow_1 = cc_out["motion"][:, 0:2, :, :]

        if frame_type == "P":
            pred = warper(shifted_ref[0], flow_1)

        elif frame_type == "B":
            flow_2 = cc_out["motion"][:, 2:4, :, :]
            beta = torch.clamp(cc_out["residue"][:, 4:5, :, :] + 0.5, 0.0, 1.0)
            pred = beta * warper(shifted_ref[0], flow_1) + (1 - beta) * warper(
                shifted_ref[1], flow_2
            )

        masked_pred = alpha * pred
        decoded_image = masked_pred + residue

    decoded_image = torch.round((2**bitdepth - 1) * decoded_image) / (2**bitdepth - 1)

    # Downsample if necessary and constraint range
    if frame_data_type == "yuv420":
        # It also constrains the range
        decoded_image = yuv_dict_clamp(convert_444_to_420(decoded_image), 0.0, 1.0)
    else:
        decoded_image = torch.clamp(decoded_image, 0.0, 1.0)

    # Quantize to 2 ** bitdepth levels
    max_dynamic = 2 ** (bitdepth) - 1
    if frame_data_type == "yuv420":
        for k, v in decoded_image.items():
            decoded_image[k] = torch.round(v * max_dynamic) / max_dynamic
    else:
        decoded_image = torch.round(decoded_image * max_dynamic) / max_dynamic

    decoded_frame = FrameData(
        bitdepth=bitdepth, frame_data_type=frame_data_type, data=decoded_image
    )

    return decoded_frame, bitstream_bytes
