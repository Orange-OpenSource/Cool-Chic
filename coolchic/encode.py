 # Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import os
import sys
import torch
import subprocess
import configargparse

from enc.bitstream.encode import encode_video
from enc.component.coolchic import CoolChicEncoderParameter
from enc.component.video import (
    VideoEncoder,
    FrameEncoderManager,
    load_video_encoder,
)
from enc.utils.codingstructure import CodingStructure
from enc.utils.misc import get_best_device


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
        "-i", "--input",
        help="Path of the input image. Either .png (RGB444) or .yuv (YUV420)",
        type=str,
    )
    parser.add(
        "-o", "--output",
        help="Path of the compressed bitstream. If empty, no bitstream is written",
        type=str,
        default="",
    )

    parser.add(
        "--workdir", help="Path of the working_directory", type=str, default="."
    )
    parser.add("--lmbda", help="Rate constraint", type=float, default=1e-3)
    parser.add_argument(
        "--job_duration_min",
        type=int,
        default=-1,
        help="Exit and save the encoding after this duration. Use -1 to only exit at the end.",
    )

    # -------- Configuration files
    parser.add(
        "--enc_cfg", is_config_file=True, help="Encoder configuration file"
    )

    parser.add(
        "--dec_cfg", is_config_file=True, help="Decoder configuration file"
    )

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

    parser.add(
        "--start_lr", help="Initial learning rate", type=float, default=1e-2
    )
    parser.add(
        "--n_itr",
        help="Maximum number of iterations per phase",
        type=int,
        default=int(1e4),
    )
    parser.add_argument(
        "--n_train_loops", help="Number of training loops", type=int, default=1
    )
    parser.add_argument(
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
        help="Syntax example for the synthesis:"
        " 12-1-linear-relu,12-1-residual-relu,X-1-linear-relu,X-3-residual-none"
        "This is a 4 layers synthesis. Now the output layer (computing the final RGB"
        "values) must be specified i.e. a 12,12 should now be called a 12,12,3. Each layer"
        "is described using the following syntax:"
        "<output_dim>-<kernel_size>-<type>-<non_linearity>. "
        "<output_dim> is the number of output features. If set to X, this is replaced by the"
        "number of required output features i.e. 3 for a RGB or YUV frame."
        "<kernel_size> is the spatial dimension of the kernel. Use 1 to mimic an MLP."
        "<type> is either 'linear' for a standard conv or 'residual' for a residual"
        " block i.e. layer(x) = x + conv(x). <non_linearity> Can be'none' for no"
        " non-linearity, 'relu' for a ReLU, 'leakyrelu' for a LeakyReLU."
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
        "--n_ft_per_res",
        type=str,
        default="1,1,1,1,1,1,1",
        help="Number of feature for each latent resolution. e.g. --n_ft_per_res=1,2,2,2,3,3,3"
        " for 7 latent grids with variable resolutions.",
    )

    parser.add(
        "--upsampling_kernel_size",
        help="upsampling kernel size (≥4 and multiple of 2)",
        type=int,
        default=8,
    )
    parser.add(
        "--static_upsampling_kernel",
        help="Use this flag to **not** learn the upsampling kernel",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    print(args)
    print("----------")
    print(parser.format_values())    # useful for logging where different settings came from
    # =========================== Parse arguments =========================== #

    # =========================== Parse arguments =========================== #
    workdir = f'{args.workdir.rstrip("/")}/'

    path_video_encoder = f"{workdir}video_encoder.pt"
    if os.path.exists(path_video_encoder):
        video_encoder = load_video_encoder(path_video_encoder)

    else:

        start_print = (
            '\n\n'
            '*----------------------------------------------------------------------------------------------------------*\n'
            '|                                                                                                          |\n'
            '|                                                                                                          |\n'
            '|       ,gggg,                                                                                             |\n'
            '|     ,88"""Y8b,                           ,dPYb,                             ,dPYb,                       |\n'
            '|    d8"     `Y8                           IP\'`Yb                             IP\'`Yb                       |\n'
            '|   d8\'   8b  d8                           I8  8I                             I8  8I      gg               |\n'
            '|  ,8I    "Y88P\'                           I8  8\'                             I8  8\'      ""               |\n'
            '|  I8\'             ,ggggg,      ,ggggg,    I8 dP      aaaaaaaa        ,gggg,  I8 dPgg,    gg     ,gggg,    |\n'
            '|  d8             dP"  "Y8ggg  dP"  "Y8ggg I8dP       """"""""       dP"  "Yb I8dP" "8I   88    dP"  "Yb   |\n'
            '|  Y8,           i8\'    ,8I   i8\'    ,8I   I8P                      i8\'       I8P    I8   88   i8\'         |\n'
            '|  `Yba,,_____, ,d8,   ,d8\'  ,d8,   ,d8\'  ,d8b,_                   ,d8,_    _,d8     I8,_,88,_,d8,_    _   |\n'
            '|    `"Y8888888 P"Y8888P"    P"Y8888P"    8P\'"Y88                  P""Y8888PP88P     `Y88P""Y8P""Y8888PP   |\n'
            '|                                                                                                          |\n'
            '|                                                                                                          |\n'
            '| version 3.2                                                                           © 2023-2024 Orange |\n'
            '*----------------------------------------------------------------------------------------------------------*\n'

        )

        print(start_print)

        subprocess.call(f"mkdir -p {workdir}", shell=True)

        # Dump raw parameters into a text file to keep track
        with open(f"{workdir}param.txt", "w") as f_out:
            f_out.write(str(args))
            f_out.write("\n")
            f_out.write("----------\n")
            f_out.write(parser.format_values())    # useful for logging where different settings came from

        # ----- Create coding configuration
        assert args.intra_period >= 0 and args.intra_period <= 255, (
            f"Intra period should be " f"  in [0, 255]. Found {args.intra_period}"
        )

        assert args.p_period >= 0 and args.p_period <= 255, (
            f"P period should be " f"  in [0, 255]. Found {args.p_period}"
        )

        is_image = (
            args.input.endswith(".png")
            or args.input.endswith(".PNG")
            or args.input.endswith(".jpeg")
            or args.input.endswith(".JPEG")
            or args.input.endswith(".jpg")
            or args.input.endswith(".JPG")
        )

        if is_image:
            assert args.intra_period == 0 and args.p_period == 0, (
                f"Encoding a PNG or JPEG image {args.input} must be done with "
                "intra_period = 0 and p_period = 0. Found intra_period = "
                f"{args.intra_period} and p_period = {args.p_period}"
            )

        coding_config = CodingStructure(
            intra_period=args.intra_period,
            p_period=args.p_period,
            seq_name=os.path.basename(args.input).split(".")[0],
        )

        # Parse arguments
        layers_synthesis = [x for x in args.layers_synthesis.split(",") if x != ""]
        n_ft_per_res = [int(x) for x in args.n_ft_per_res.split(",") if x != ""]

        assert set(n_ft_per_res) == {1}, (
            f"--n_ft_per_res should only contains 1. Found {args.n_ft_per_res}"
        )

        assert len(args.arm.split(",")) == 2, (
            f"--arm format should be X,Y." f" Found {args.arm}"
        )

        dim_arm, n_hidden_layers_arm = [int(x) for x in args.arm.split(",")]

        coolchic_encoder_parameter = CoolChicEncoderParameter(
            layers_synthesis=layers_synthesis,
            dim_arm=dim_arm,
            n_hidden_layers_arm=n_hidden_layers_arm,
            n_ft_per_res=n_ft_per_res,
            upsampling_kernel_size=args.upsampling_kernel_size,
            static_upsampling_kernel=args.static_upsampling_kernel,
        )

        frame_encoder_manager = FrameEncoderManager(
            preset_name=args.recipe,
            start_lr=args.start_lr,
            lmbda=args.lmbda,
            n_loops=args.n_train_loops,
            n_itr=args.n_itr,
        )

        video_encoder = VideoEncoder(
            coding_structure=coding_config,
            shared_coolchic_parameter=coolchic_encoder_parameter,
            shared_frame_encoder_manager=frame_encoder_manager,
        )

    # Automatic device detection
    device = get_best_device()
    print(f'{"Device":<20}: {device}')

    # # ====================== Torchscript JIT parameters ===================== #
    # # From https://github.com/pytorch/pytorch/issues/52286
    # # This is no longer the case with the with torch.jit.fuser
    # # ! This gives a significant (+25 %) speed up
    # torch._C._jit_set_profiling_executor(False)
    # torch._C._jit_set_texpr_fuser_enabled(False)
    # torch._C._jit_set_profiling_mode(False)

    # torch.set_float32_matmul_precision("high")
    # # ====================== Torchscript JIT parameters ===================== #

    if device == "cpu":
        # the number of cores is adjusted wrt to the slurm variable if exists
        n_cores = os.getenv("SLURM_JOB_CPUS_PER_NODE")
        # otherwise use the machine cpu count
        if n_cores is None:
            n_cores = os.cpu_count()

        n_cores = int(n_cores)
        print(f'{"CPU cores":<20}: {n_cores}')

    elif device == "cuda:0":
        # ! This one makes the training way faster!
        torch.backends.cudnn.benchmark = True

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
    if args.output != "":
        # video_encoder = load_video_encoder(video_encoder_savepath)
        encode_video(video_encoder, args.output, hls_sig_blksize=16)

    sys.exit(exit_code.value)

