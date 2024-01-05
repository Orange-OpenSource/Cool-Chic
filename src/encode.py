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
import argparse
from bitstream.encode import encode_video

from encoding_management.coding_structure import CodingStructure
from models.coolchic_encoder import CoolChicEncoderParameter
from models.frame_encoder import FrameEncoderManager
from models.video_encoder import VideoEncoder, load_video_encoder
from utils.misc import get_best_device


"""
Use this file to train i.e. encode a GOP i.e. something which starts with one
intra frame and is then followed by <intra_period> inter frames. Note that an
image is simply a GOP of size 1 with no inter frames.

Input can accommodate either .png (single-frame GOP) or .yuv file (1 to 256
frames in the GOP).

Syntax example for the synthesis:
    12-1-linear,12-1-residual,X-1-linear,X-3-residual

    This is a 4 layers synthesis. Now the output layer (computing the final RGB
values) must be specified i.e. a 12,12 should now be called a 12,12,3. Each layer
is described using the following syntax:

        <output_dim>-<kernel_size>-<type>-<non_linearity>

    <output_dim>    : Number of output features. If set to X, this is replaced by the
                      number of required output features i.e. 3 for a RGB or YUV frame.
    <kernel_size>   : Spatial dimension of the kernel. Use 1 to mimic an MLP.
    <type>          :   - "linear" for a standard conv,
                        - "residual" for a residual block i.e. layer(x) = x + conv(x)
                        - "attention" for an attention block i.e.
                            layer(x) = x + conv_1(x) * sigmoid(conv_2(x))
    <non_linearity> :   - "none" for no non-linearity,
                        - "relu" for a ReLU,
                        - "leakyrelu" for a LeakyReLU,
                        - "gelu" for a GELU,
"""

if __name__ == '__main__':
    # =========================== Parse arguments =========================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path of the input image. Either .png (RGB444) or .yuv (YUV420)', type=str)
    parser.add_argument('-o', '--output', type=str, default='./bitstream.bin', help='Output bitstream path.')
    parser.add_argument('--workdir', help='Path of the working_directory', type=str, default='.')

    parser.add_argument('--lmbda', help='Rate constraint', type=float, default=1e-3)
    parser.add_argument('--start_lr', help='Initial learning rate', type=float, default=1e-2)
    parser.add_argument('--n_itr', help='Maximum number of iterations per phase', type=int, default=int(1e5))
    parser.add_argument(
        '--layers_synthesis', help='See default.', type=str,
        default='40-1-linear-relu,X-1-linear-relu,X-3-residual-relu,X-3-residual-none'
    )
    parser.add_argument('--layers_arm', help='Format: 16,8,16', type=str, default='24,24')
    parser.add_argument('--dist', help='Unused for now', type=str, default='mse')
    parser.add_argument('--n_ctx_rowcol', help='Number of rows/columns for ARM', type=int, default=3)
    parser.add_argument(
        '--n_ft_per_res', type=str, default='1,1,1,1,1,1,1',
        help='Number of feature for each latent resolution. e.g. --n_ft_per_res=1,2,2,2,3,3,3'
        ' for 7 latent grids with variable resolutions.',
    )

    parser.add_argument('--upsampling_kernel_size', help='upsampling kernel size (≥8)', type=int, default=8)
    parser.add_argument('--n_train_loops', help='Number of training loops', type=int, default=5)
    args = parser.parse_args()
    # =========================== Parse arguments =========================== #

    # ====================== Torchscript JIT parameters ===================== #
    # From https://github.com/pytorch/pytorch/issues/52286
    # This is no longer the case with the with torch.jit.fuser
    # ! This gives a significant (+25 %) speed up
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_profiling_mode(False)
    # ====================== Torchscript JIT parameters ===================== #

    # =========================== Parse arguments =========================== #
    workdir = f'{args.workdir.rstrip("/")}/'
    subprocess.call(f'mkdir -p {workdir}', shell=True)

    # Dump raw parameters into a text file to keep track
    with open(f'{workdir}param.txt', 'w') as f_out:
        f_out.write(str(sys.argv))

    # Parse arguments
    layers_synthesis = [x for x in args.layers_synthesis.split(',') if x != '']
    layers_arm = [int(x) for x in args.layers_arm.split(',') if x != '']
    n_ft_per_res = [int(x) for x in args.n_ft_per_res.split(',') if x != '']

    # Automatic device detection
    device = get_best_device()
    print(f'{"Device":<20}: {device}')
    # =========================== Parse arguments =========================== #

    # ====================== Torchscript JIT parameters ===================== #
    # From https://github.com/pytorch/pytorch/issues/52286
    # This is no longer the case with the with torch.jit.fuser
    # ! This gives a significant (+25 %) speed up
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_profiling_mode(False)

    if device == 'cpu':
        # the number of cores is adjusted wrt to the slurm variable if exists
        n_cores=os.getenv('SLURM_JOB_CPUS_PER_NODE')
        # otherwise use the machine cpu count
        if n_cores is None:
            n_cores = os.cpu_count()

        n_cores=int(n_cores)
        print(f'{"CPU cores":<20}: {n_cores}')

        torch.set_flush_denormal(True)
        # This is ignored due to the torch._C.jit instructions above
        # torch.jit.enable_onednn_fusion(True)
        torch.set_num_interop_threads(n_cores) # Inter-op parallelism
        torch.set_num_threads(n_cores) # Intra-op parallelism

        subprocess.call('export OMP_PROC_BIND=spread', shell=True)  # ! VERY IMPORTANT
        subprocess.call('export OMP_PLACES=threads', shell=True)
        subprocess.call('export OMP_SCHEDULE=static', shell=True)   # ! VERY IMPORTANT

        subprocess.call(f'export OMP_NUM_THREADS={n_cores}', shell=True)
        subprocess.call('export KMP_HW_SUBSET=1T', shell=True)
    # ====================== Torchscript JIT parameters ===================== #


    path_video_encoder = f'{workdir}video_encoder.pt'
    if os.path.exists(path_video_encoder):
        video_encoder = load_video_encoder(path_video_encoder)

    else:
        # ----- Create coding configuration
        # Create an all-intra GOP with one frame.
        coding_config = CodingStructure(
            gop_type='RA',
            intra_period=0,
            p_period=0,
            seq_name=os.path.basename(args.input).split('.')[0]
        )

        coolchic_encoder_parameter = CoolChicEncoderParameter(
            layers_synthesis=layers_synthesis,
            layers_arm=layers_arm,
            n_ctx_rowcol=args.n_ctx_rowcol,
            n_ft_per_res=n_ft_per_res,
            upsampling_kernel_size=args.upsampling_kernel_size,
        )

        dist_weight = {'mse': 1.0, 'msssim': 0.0, 'lpips': 0.0}

        frame_encoder_manager = FrameEncoderManager(
            preset_name='c3',
            start_lr=args.start_lr,
            lmbda=args.lmbda,
            n_loops=args.n_train_loops,
            dist_weight=dist_weight,
            n_itr=args.n_itr,
        )

        video_encoder = VideoEncoder(
            coding_structure=coding_config,
            shared_coolchic_parameter=coolchic_encoder_parameter,
            shared_frame_encoder_manager=frame_encoder_manager,
            path_original_sequence=args.input,
        )

    print(f'\n{video_encoder.coding_structure.pretty_string()}\n')

    # Learn the encoder
    video_encoder.train(device=device, workdir=workdir)

    # Print the final results
    print('\nFinal encoder-side results:')
    print('----------------------------')
    path_res_file = f'{workdir}/results_best.tsv'
    if os.path.isfile(path_res_file):
        print('\n'.join(open(path_res_file, 'r').readlines()))
    else:
        print(f'Can not find a final result file at {path_res_file}')

    # Encode the video into a binary file
    print(f'Encoding to {args.output}')
    encode_video(video_encoder, args.output)
