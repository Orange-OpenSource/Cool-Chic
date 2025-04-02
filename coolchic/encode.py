# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import os
import sys

import configargparse
from enc.component.coolchic import CoolChicEncoderParameter
from enc.component.frame import load_frame_encoder
from enc.component.video import (
    FrameEncoderManager,
    _get_frame_path_prefix,
    encode_one_frame,
)
from enc.utils.codingstructure import CodingStructure
from enc.utils.parsecli import (
    get_coding_structure_from_args,
    get_coolchic_param_from_args,
    get_manager_from_args,
)
import torch

if __name__ == "__main__":
    # =========================== Parse arguments =========================== #
    # By increasing priority order, the arguments work as follows:
    #
    #   1. Default value of --dummy_arg=42 is the base value.
    #
    #   2. If dummy_arg is present in either the encoder configuration file
    #     (--enc_cfg) or the decoder configuration file (--dec_cfg), then it
    #     overrides the default value.
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
    )
    parser.add(
        "-o",
        "--output",
        help="Path of the compressed bitstream. If empty, no bitstream is written",
        type=str,
        default="",
    )

    parser.add("--workdir", help="Path of the working_directory", type=str, default=".")
    parser.add("--lmbda", help="Rate constraint", type=float, default=1e-3)
    parser.add(
        "--job_duration_min",
        type=int,
        default=-1,
        help="Exit and save the encoding after this duration. Use -1 to only exit at the end.",
    )

    parser.add_argument(
        "--print_detailed_archi",
        action="store_true",
        help="Print detailed NN architecture",
    )
    parser.add_argument(
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
    parser.add("--enc_cfg", is_config_file=True, help="Encoder config file")
    parser.add(
        "--dec_cfg_residue",
        is_config_file=True,
        help="Residue (or intra) decoder config file",
    )
    parser.add(
        "--dec_cfg_motion", is_config_file=True, help="Motion decoded config file"
    )

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

    parser.add("--n_train_loops", help="Number of training loops", type=int, default=1)
    parser.add(
        "--preset",
        help='Recipe type. Either "c3x" or "debug".',
        type=str,
        default="c3x",
    )

    # ==== Decoder-side arguments
    parser.add(
        "--layers_synthesis_residue",
        type=str,
        default="48-1-linear-relu,X-1-linear-none,X-3-residual-relu,X-3-residual-none",
        help="Syntax example: "
        " "
        "    12-1-linear-relu,12-1-residual-relu,X-1-linear-relu,X-3-residual-none "
        " "
        "    This is a 4 layers synthesis. Each layer is separated by comas and is "
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
        "with a residual connexion block i.e. layer(x) = x + conv(x). "
        " "
        "<non_linearity> Can be 'none' for no non-linearity, 'relu' for a ReLU "
        "non-linearity. "
        " Parameterize the residue decoder.",
    )
    parser.add(
        "--layers_synthesis_motion",
        type=str,
        default="9-1-linear-relu,X-1-linear-none,X-3-residual-none",
        help="Identical to --layers_synthesis_residue but for the motion decoder.",
    )

    parser.add(
        "--arm_residue",
        type=str,
        default="16,2",
        help="<arm_context_and_layer_dimension>,<number_of_hidden_layers> "
        "First number indicates both the context size **AND** the hidden layer dimension. "
        "Second number indicates the number of hidden layer(s). 0 gives a linear ARM module. "
        " Parameterize the residue decoder.",
    )
    parser.add(
        "--arm_motion",
        type=str,
        default="8,1",
        help="Identical to --arm_residue but for the motion decoder.",
    )

    parser.add(
        "--n_ft_per_res_residue",
        type=str,
        default="1,1,1,1,1,1,1",
        help="Number of feature for each latent resolution. e.g. "
        " --n_ft_per_res_residue=1,1,1,1,1,1,1 "
        " for 7 latent grids with variable resolutions. "
        " Parameterize the residue decoder.",
    )
    parser.add(
        "--n_ft_per_res_motion",
        type=str,
        default="1,1,1,1,1,1,1",
        help="Identical to --n_ft_per_res_residue but for the motion decoder.",
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
        "--n_ft_per_res",
        type=str,
        default="1,1,1,1,1,1,1",
        help="Number of feature for each latent resolution. e.g. "
        " --n_ft_per_res_residue=1,1,1,1,1,1,1 "
        " for 7 latent grids with variable resolutions. "
        " Parameterize the residue decoder.",
    )

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
    if os.path.exists(path_frame_encoder):
        frame_encoder, frame_encoder_manager = load_frame_encoder(path_frame_encoder)
        coolchic_enc_param = frame_encoder.coolchic_enc_param

    else:
        start_print = (
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
            "| version 4.0.0, March 2025                                                             © 2023-2025 Orange |\n"
            "*----------------------------------------------------------------------------------------------------------*\n"
        )
        print(start_print)

        # Dump raw parameters into a text file to keep track
        with open(f"{frame_save_prefix}param.txt", "w") as f_out:
            f_out.write(str(args))
            f_out.write("\n")
            f_out.write("----------\n")
            f_out.write(
                parser.format_values()
            )  # useful for logging where different settings came from

        # Successively parse the Cool-chic architectures for the residue
        # and the motion Cool-chic
        coolchic_enc_param = {
            cc_name: CoolChicEncoderParameter(
                **get_coolchic_param_from_args(args, cc_name)
            )
            for cc_name in ["residue", "motion"]
        }
        frame_encoder_manager = FrameEncoderManager(**get_manager_from_args(args))

    # Automatic device detection
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"{'Device':<26}: {device}")

    print(
        f"\n{coding_structure.pretty_string(print_detailed_struct=args.print_detailed_struct)}\n"
    )

    exit_code = encode_one_frame(
        video_path=args.input,
        coding_structure=coding_structure,
        coolchic_enc_param=coolchic_enc_param,
        frame_encoder_manager=frame_encoder_manager,
        coding_index=args.coding_idx,
        job_duration_min=args.job_duration_min,
        device=device,
        print_detailed_archi=args.print_detailed_archi,
    )

    # Bitstream
    if args.output != "":
        from enc.bitstream.encode import encode_frame

        frame_enc_path = (
            f"{_get_frame_path_prefix(frame.display_order)}frame_encoder.pt"
        )
        frame_encoder, _ = load_frame_encoder(frame_enc_path)
        encode_frame(
            frame_encoder, None, args.output, frame.display_order, hls_sig_blksize=16
        )

    sys.exit(exit_code.value)
