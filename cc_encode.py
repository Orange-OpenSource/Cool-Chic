# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import os
import sys
import typing

import configargparse
import torch

from coolchic.component.core.coolchic import CoolChicEncoderParameter
from coolchic.component.frame import load_frame_encoder
from coolchic.component.intercoding.warp import WarpParameter
from coolchic.component.video import (
    _get_frame_path_prefix,
    encode_one_frame,
)
from coolchic.io.io import load_frame_data_from_file
from coolchic.training.loss import DISTORTION_METRIC, loss_function
from coolchic.training.presets import AVAILABLE_PRESETS
from coolchic.utils.codingstructure import CodingStructure
from coolchic.utils.parsecli import (
    get_coding_structure_from_args,
    get_coolchic_param_from_args,
    get_preset_from_args,
    get_warp_param_from_args,
)

if __name__ == "__main__":
    # =========================== Parse arguments =========================== #
    # By increasing priority order, the arguments work as follows:
    #
    #   1. Default value of --dummy_arg=42 is the base value.
    #
    #   2. If dummy_arg is present in  the decoder configuration file (--dec_cfg),
    #     then it overrides the default value.
    #
    #   3. If --dummy_arg is explicitly provided in the command line, then it
    #      overrides both the default value and the value listed in the
    #      configuration file.

    parser = configargparse.ArgumentParser()
    # -------- These arguments are not in the configuration files
    parser.add(
        "-i",
        "--input",
        help="Path of the input image. Either .png or .yuv",
        type=str,
        required=True,
    )
    parser.add(
        "-o",
        "--output",
        help="Path of the compressed bitstream.",
        type=str,
        default="./bitstream.cool",
    )
    parser.add("--nobitstream", action="store_true", help="Don't write a bitstream")

    parser.add("--workdir", help="Path of the working_directory", type=str, default=".")
    parser.add("--lmbda", help="Rate constraint", type=float, default=1e-3)

    parser.add(
        "--print_detailed_archi",
        action="store_true",
        help="Print detailed NN architecture",
    )
    parser.add(
        "--print_detailed_struct",
        action="store_true",
        help="Print detailed Coding structure",
    )

    parser.add(
        "--intra_pos",
        help="Display index of the intra frames. "
        "Format is 0,4,7 if you want the frame 0, 4 and 7 to be intra frames. "
        "-1 can be used to denote the last frame, -2 the 2nd to last etc. "
        "x-y is a range from x (included) to y (included). This does not work "
        "with the negative indexing. "
        "0,4-7,-2 ==> Intra for the frame 0, 4, 5, 6, 7 and the 2nd to last."
        "Frame 0 must be an intra frame.",
        type=str,
        default="0",
    )
    parser.add(
        "--p_pos",
        help="Display index of the P frames. Same format than --intra_pos ",
        type=str,
        default="",
    )
    parser.add(
        "--n_frames",
        help="How many frames to code",
        type=int,
        default=1,
    )
    parser.add(
        "--frame_offset",
        help="Shift the position of the 0-th frame of the video. "
        "If --frame_offset=15 skip the first 15 frames of the video.",
        type=int,
        default=0,
    )

    parser.add(
        "--coding_idx",
        help="Index (in coding order) of the frame to be coded. 0 is first",
        type=int,
        default=0,
    )

    # -------- Configuration files
    parser.add(
        "--dec_cfg_residue",
        is_config_file=True,
        help="Residue (or intra) decoder config file",
    )
    parser.add("--dec_cfg_motion", is_config_file=True, help="Motion decoded config file")

    # -------- These arguments are in the configuration files

    # ==== Encoder-side arguments

    parser.add("--start_lr", help="Initial learning rate", type=float, default=1e-2)
    parser.add(
        "--n_itr",
        help="Number of iterations for the main training stage. Use this to "
        "change the duration of the encoding.",
        type=int,
        default=int(1e4),
    )
    parser.add(
        "--n_itr_pretrain_motion",
        help="Number of iterations for the motion pre-training stage.",
        type=int,
        default=int(1e3),
    )

    parser.add(
        "--tune",
        help="Preset used to perform subjective tuning. Available: "
        f"{typing.get_args(DISTORTION_METRIC)}",
        type=str,
        default="mse",
        choices=typing.get_args(DISTORTION_METRIC),
    )

    parser.add("--debug", action="store_true", help="Extremely quick training")

    # ==== Decoder-side arguments
    parser.add(
        "--layers_synthesis_residue",
        type=str,
        default="48-1-linear-relu,X-1-linear-none,X-3-residual-relu,X-3-residual-none/stabiliser",
        help="Syntax example: "
        " "
        "    12-1-linear-relu,12-1-residual-relu,X-1-linear-relu,X-3-residual-none/stabiliser "
        " "
        "   This is a 4 layers synthesis with a stabiliser layer activated in parallel. See "
        " enc/component/core/synthesis.py for more info. Remove '/stabiliser' if no stabiliser "
        " layer should be used."
        " "
        "   Each layer is separated by comas and is "
        "described using the following syntax: "
        " "
        "    <output_dim>-<kernel_size>-<type>-<non_linearity>. "
        " "
        "<output_dim> is the number of output features. If set to X, this is replaced by the "
        "number of required output features i.e. 3 for a RGB or YUV frame or 4 for a P-frame "
        "residue + alpha. "
        " "
        "<kernel_size> is the spatial dimension of the kernel. Use 1 to mimic an MLP. "
        " "
        "<type> is either 'linear' for a standard conv or 'residual' for a convolution "
        "with a residual connection block i.e. layer(x) = x + conv(x). "
        " "
        "<non_linearity> Can be 'none' for no non-linearity, 'relu' for a ReLU "
        "non-linearity. "
        " Parameterize the residue decoder.",
    )
    parser.add(
        "--layers_synthesis_motion",
        type=str,
        default="16-1-linear-relu,X-1-linear-none/stabiliser",
        help="Identical to --layers_synthesis_residue but for the motion decoder.",
    )

    parser.add(
        "--arm_residue",
        type=str,
        default="14,2/stabiliser",
        help="<arm_context_and_layer_dimension>,<number_of_hidden_layers>/stabiliser "
        "First number indicates both the context size **AND** the hidden layer dimension. "
        "Second number indicates the number of hidden layer(s). 0 gives a linear ARM module. "
        " Parameterize the residue decoder."
        "Remove '/stabiliser' if no stabiliser "
        " layer should be used.",
    )
    parser.add(
        "--arm_motion",
        type=str,
        default="6,2/stabiliser",
        help="Identical to --arm_residue but for the motion decoder.",
    )

    parser.add(
        "--output_feature_ifce_residue",
        type=int,
        default=6,
        help="<number of outputs>"
        "The number of output features extracted by the inter features ARM. "
        "Note: All inter feature arm are disabled if sets to 0.",
    )

    parser.add(
        "--output_feature_ifce_motion",
        type=int,
        default=6,
        help="Identical to --output_feature_ifce_residue but for the motion decoder.",
    )

    parser.add(
        "--ifce_resolution_residue",
        type=str,
        default="0-2",
        help="<start resolution reduction>-<end resolution reduction>"
        "The inclusive resolution reduction range where inter feature arm are enabled."
        ""
        "Syntax example: "
        ""
        "  1-4"
        ""
        "  This activates inter feature arm for levels corresponding to 1/1 original resolution down"
        "  to 1/4 original resolution."
        ""
        "Set to 'no' to disable all inter feature arm."
        ""
        "Note: enabled hyperlatents on the included range will also have an inter feature arm.",
    )

    parser.add(
        "--ifce_resolution_motion",
        type=str,
        default="2-2",
        help="Identical to --ifce_resolution_residue but for the motion decoder.",
    )

    parser.add(
        "--hyperlatent_resolution_residue",
        type=str,
        default="auto",
        help="<start resolution reduction>-<end resolution reduction>"
        "The inclusive resolution reduction range where hyperlatents are enabled."
        ""
        "Syntax example: "
        ""
        "  4-6"
        ""
        "  This activates hyperlatents for latents corresponding to 1/2^-4 original resolution down"
        "  to 1/2^-6 original resolution."
        ""
        "Set to 'auto' to automatically set the range depending on the number of latents."
        "Set to 'no' to disable all hyperlatents.",
    )

    parser.add(
        "--hyperlatent_resolution_motion",
        type=str,
        default="no",
        help="Identical to --hyperlatent_resolution_residue but for the motion decoder.",
    )

    parser.add(
        "--latent_resolution_residue",
        type=str,
        default="auto",
        help="<start resolution>-<end resolution>"
        "The inclusive resolution range of the latent grid."
        ""
        "Syntax example: "
        ""
        "  0-6"
        ""
        "  This gives latents grid from the 1/2^0 = 1/1 original resolution down"
        "  to 1/2^6 = 1/64 original resolution."
        ""
        "Set to 'auto' to automatically set the range depending on the image size.",
    )

    parser.add(
        "--latent_resolution_motion",
        type=str,
        default="2-6",
        help="Identical to --latent_resolution_residue but for the motion decoder.",
    )

    parser.add(
        "--ups_k_size_residue",
        type=int,
        default=8,
        help="Upsampling kernel size for the transposed convolutions. "
        "Must be even and >= 4. Parameterize the residue decoder.",
    )
    parser.add(
        "--ups_k_size_motion",
        type=int,
        default=8,
        help="Identical to --ups_k_size_residue but for the motion decoder.",
    )

    parser.add(
        "--ups_preconcat_k_size_residue",
        type=int,
        default=7,
        help="Upsampling kernel size for the pre-concatenation convolutions. "
        "Must be odd. Parameterize the residue decoder.",
    )
    parser.add(
        "--ups_preconcat_k_size_motion",
        type=int,
        default=7,
        help="Identical to --ups_preconcat_k_size_residue but for the motion decoder.",
    )

    parser.add(
        "--warp_filter_size",
        type=int,
        default=8,
        help="Number of taps for the warping interpolation filter. ",
    )

    # display the GPU type and capabilities
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_properties(i).name
        total_memory = (
            torch.cuda.get_device_properties(i).total_memory / 1024.0 / 1024.0
        )  # expressed in bytes --> MBytes
        print(f"GPU: {name:20s} Mem: {total_memory:.0f}")

    args = parser.parse_args()
    print(args)
    print("----------")
    # useful for logging where different settings came from
    print(parser.format_values())
    # =========================== Parse arguments =========================== #

    # =========================== Parse arguments =========================== #
    os.chdir(args.workdir)

    # This is not stored inside a FrameEncoder so we need to reconstruct it
    coding_structure = CodingStructure(**get_coding_structure_from_args(args))

    # Get the frame to code
    frame = coding_structure.get_frame_from_coding_order(args.coding_idx)

    # Check if we have some 000X-frame_encoder.pt somewhere
    frame_save_prefix = _get_frame_path_prefix(frame.display_order)
    path_frame_encoder = f"{frame_save_prefix}frame_encoder.pt"
    print(
        "\n\n"
        "*----------------------------------------------------------------------------------------------------------*\n"
        "|                                                                                                          |\n"
        "|                                                                                                          |\n"
        "|       ,gggg,                                                                                             |\n"
        '|     ,88"""Y8b,                           ,dPYb,                             ,dPYb,                       |\n'
        "|    d8\"     `Y8                           IP'`Yb                             IP'`Yb                       |\n"
        "|   d8'   8b  d8                           I8  8I                             I8  8I      gg               |\n"
        "|  ,8I    \"Y88P'                           I8  8'                             I8  8'      \"\"               |\n"
        "|  I8'             ,ggggg,      ,ggggg,    I8 dP      aaaaaaaa        ,gggg,  I8 dPgg,    gg     ,gggg,    |\n"
        '|  d8             dP"  "Y8ggg  dP"  "Y8ggg I8dP       """"""""       dP"  "Yb I8dP" "8I   88    dP"  "Yb   |\n'
        "|  Y8,           i8'    ,8I   i8'    ,8I   I8P                      i8'       I8P    I8   88   i8'         |\n"
        "|  `Yba,,_____, ,d8,   ,d8'  ,d8,   ,d8'  ,d8b,_                   ,d8,_    _,d8     I8,_,88,_,d8,_    _   |\n"
        '|    `"Y8888888 P"Y8888P"    P"Y8888P"    8P\'"Y88                  P""Y8888PP88P     `Y88P""Y8P""Y8888PP   |\n'
        "|                                                                                                          |\n"
        "|                                                                                                          |\n"
        "| version 5.0.1, June 2026                                                              © 2023-2026 Orange |\n"
        "*----------------------------------------------------------------------------------------------------------*\n"
    )

    # Dump raw parameters into a text file to keep track
    with open(f"{frame_save_prefix}param.txt", "w") as f_out:
        f_out.write(f"{str(args)}\n\n{parser.format_values()}")

    # Successively parse the Cool-chic architectures for the residue
    # and the motion Cool-chic
    coolchic_enc_param = {
        cc_name: CoolChicEncoderParameter(**get_coolchic_param_from_args(args, cc_name))
        for cc_name in ["residue", "motion"]
    }

    warp_parameter = WarpParameter(**get_warp_param_from_args(args))

    preset_parameter = get_preset_from_args(args)
    if args.debug:
        preset_name = "debug"
    else:
        preset_name = "intra" if frame.frame_type == "I" else "inter"
    preset = AVAILABLE_PRESETS.get(preset_name)(**preset_parameter)

    if os.path.exists(path_frame_encoder):
        next_frame = coding_structure.get_frame_from_coding_order(frame.coding_order + 1)
        msg = (
            f"Frame {frame.frame_type}{frame.display_order} is already done. "
            f"The resulting frame_encoder.pt is at "
            f"{os.path.join(args.workdir, path_frame_encoder)}"
        )
        if next_frame is not None:
            msg += (
                f"\nHint: you can now encode frame {next_frame.frame_type}{next_frame.display_order}"
                f" using --coding_idx={next_frame.coding_order}"
            )
        print(msg)

    else:
        # Automatic device detection
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"{'Device':<26}: {device}")
        if device == "cuda:0":
            torch.cuda.memory.reset_peak_memory_stats()

        print(
            f"\n{coding_structure.pretty_string(print_detailed_struct=args.print_detailed_struct)}\n"
        )

        encode_one_frame(
            video_path=args.input,
            coding_structure=coding_structure,
            coolchic_enc_param=coolchic_enc_param,
            warp_parameter=warp_parameter,
            coding_index=args.coding_idx,
            training_preset=preset,
            device=device,
        )
        if device == "cuda:0":
            print(
                f"Maximum GPU memory allocated [GBytes]: "
                f"{torch.cuda.memory.max_memory_allocated() / 1e9:6.2f}"
            )

    # Bitstream
    if not args.nobitstream:
        from coolchic.bitstream.decode import decode_video
        from coolchic.bitstream.encode import encode_frame

        frame_enc_path = f"{_get_frame_path_prefix(frame.display_order)}frame_encoder.pt"
        frame_encoder = load_frame_encoder(frame_enc_path)
        print(f"\nWriting bitstream to {args.output}")
        bytes_frame = encode_frame(frame_encoder, args.output, coding_structure)

        name, ext = os.path.splitext(os.path.basename(args.input))
        results_log_path = f"{_get_frame_path_prefix(frame.display_order)}results_decoder.tsv"
        print(f"\nDecoding the bitstream, logging the results into: {results_log_path}")

        all_decoded_frame = decode_video(
            args.output,
            decoded_path=None,
            max_decoding_order=frame.coding_order,
        )
        decoded_image = all_decoded_frame[str(frame.display_order)]

        target_image = load_frame_data_from_file(args.input, frame.display_order)

        with torch.no_grad():
            logs = loss_function(
                decoded_image=decoded_image.data,
                target_image=target_image.data,
                rate_latent_bit={"total_rate": torch.tensor(len(bytes_frame) * 8.0)},
                dist_weight=preset.dist_weight,
                lmbda=preset.lmbda,
                compute_logs=True,
            )
        if decoded_image.frame_data_type == "yuv420":
            h, w = decoded_image.data.get("y").size()[-2:]
        else:
            h, w = decoded_image.data.size()[-2:]

        results = {
            "loss": f"{logs.loss.item() * 1000.0:9.7f}",
            "psnr_db": f"{logs.detailed_dist_db.get('psnr_db'):9.7f}",
            "rate_bpp": f"{logs.total_rate_bpp:9.7f}",
            "lmbda": f"{preset.lmbda}",
            "seq_name": f"{coding_structure.seq_name}",
            "n_pixels": f"{h * w}",
            "display_order": f"{frame.display_order}",
            "coding_order": f"{frame.coding_order}",
        }

        header_str = ""
        value_str = ""
        for k, v in results.items():
            col_width = max(len(k), len(v)) + 2
            header_str += f"{k:<{col_width}} "
            value_str += f"{v:<{col_width}} "

        print("\n" + header_str + "\n" + value_str)
        with open(results_log_path, "w") as f_out:
            f_out.write(header_str + "\n" + value_str + "\n")

    sys.exit(0)
