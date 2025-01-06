# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import os
import subprocess
import sys

import configargparse
from enc.component.coolchic import CoolChicEncoderParameter
from enc.component.video import (
    FrameEncoderManager,
    VideoEncoder,
    load_video_encoder,
)
from enc.utils.codingstructure import CodingStructure
from enc.utils.misc import TrainingExitCode, get_best_device
from enc.utils.parsecli import (
    get_coding_structure_from_args,
    get_coolchic_param_from_args,
    get_manager_from_args,
)

"""
Use this file to train i.e. encode a GOP i.e. something which starts with one
intra frame and is then followed by <intra_period> inter frames. Note that an
image is simply a GOP of size 1 with no inter frames.
"""

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
        help="Path of the input image. Either .png (RGB444) or .yuv (YUV420)",
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

    # -------- Configuration files
    parser.add("--enc_cfg", is_config_file=True, help="Encoder configuration file")

    parser.add("--dec_cfg", is_config_file=True, help="Decoder configuration file")

    # -------- These arguments are in the configuration files

    # ==== Encoder-side arguments
    parser.add(
        "--intra_period",
        help="Number of inter frames in the GOP. 0 for image coding",
        type=int,
        default=0,
    )
    parser.add(
        "--p_period",
        help="Distance between P-frames. 0 for image coding",
        type=int,
        default=0,
    )

    parser.add("--start_lr", help="Initial learning rate", type=float, default=1e-2)
    parser.add(
        "--n_itr",
        help="Maximum number of iterations per phase",
        type=int,
        default=int(1e4),
    )
    parser.add("--n_train_loops", help="Number of training loops", type=int, default=1)
    parser.add(
        "--recipe",
        help='Recipe type. Either "c3x" or "debug".',
        type=str,
        default="c3x",
    )

    # ==== Encoder-side arguments
    parser.add(
        "--layers_synthesis",
        type=str,
        default="40-1-linear-relu,X-1-linear-none,X-3-residual-relu,X-3-residual-none",
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
        "number of required output features i.e. 3 for a RGB or YUV frame. "
        " "
        "<kernel_size> is the spatial dimension of the kernel. Use 1 to mimic an MLP. "
        " "
        "<type> is either 'linear' for a standard conv or 'residual' for a convolution "
        "with a residual connexion block i.e. layer(x) = x + conv(x). "
        " "
        "<non_linearity> Can be 'none' for no non-linearity, 'relu' for a ReLU "
        "non-linearity. ",
    )

    parser.add(
        "--arm",
        type=str,
        default="24,2",
        help="<arm_context_and_layer_dimension>,<number_of_hidden_layers>"
        "First number indicates both the context size **AND** the hidden layer dimension."
        "Second number indicates the number of hidden layer(s). 0 gives a linear ARM module.",
    )

    parser.add(
        "--ups_k_size",
        type=int,
        default=8,
        help="Upsampling kernel size for the transposed convolutions. "
        "Must be even and >= 4.",
    )

    parser.add(
        "--ups_preconcat_k_size",
        type=int,
        default=7,
        help="Upsampling kernel size for the pre-concatenation convolutions. "
        "Must be odd.",
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
    print(
        parser.format_values()
    )  # useful for logging where different settings came from
    # =========================== Parse arguments =========================== #

    # =========================== Parse arguments =========================== #
    workdir = f'{args.workdir.rstrip("/")}/'

    path_video_encoder = f"{workdir}video_encoder.pt"
    if os.path.exists(path_video_encoder):
        video_encoder = load_video_encoder(path_video_encoder)

    else:
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
            "| version 3.4.1, Jan. 2025                                                              © 2023-2025 Orange |\n"
            "*----------------------------------------------------------------------------------------------------------*\n"
        )

        subprocess.call(f"mkdir -p {workdir}", shell=True)

        # Dump raw parameters into a text file to keep track
        with open(f"{workdir}param.txt", "w") as f_out:
            f_out.write(str(args))
            f_out.write("\n")
            f_out.write("----------\n")
            f_out.write(
                parser.format_values()
            )  # useful for logging where different settings came from

        # ----- Parse arguments & construct video encoder
        coding_structure = CodingStructure(**get_coding_structure_from_args(args))
        coolchic_encoder_parameter = CoolChicEncoderParameter(
            **get_coolchic_param_from_args(args)
        )
        frame_encoder_manager = FrameEncoderManager(**get_manager_from_args(args))

        video_encoder = VideoEncoder(
            coding_structure=coding_structure,
            shared_coolchic_parameter=coolchic_encoder_parameter,
            shared_frame_encoder_manager=frame_encoder_manager,
        )

    # Automatic device detection
    device = get_best_device()
    print(f'{"Device":<20}: {device}')

    print(f"\n{video_encoder.coding_structure.pretty_string()}\n")
    exit_code = video_encoder.encode(
        path_original_sequence=args.input,
        device=device,
        workdir=workdir,
        job_duration_min=args.job_duration_min,
    )

    video_encoder_savepath = f"{workdir}video_encoder.pt"
    video_encoder.save(video_encoder_savepath)

    # Bitstream
    if args.output != "" and exit_code == TrainingExitCode.END:
        from enc.bitstream.encode import encode_video
        encode_video(video_encoder, args.output, hls_sig_blksize=16)

    sys.exit(exit_code.value)
