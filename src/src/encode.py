# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
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

from PIL import Image
from torchvision.transforms.functional import to_tensor

from bitstream.encode import encode
from bitstream.decode import decode
from model_management.io import save_model

from model_management.presets import AVAILABLE_PRESET
from model_management.trainer import do_warmup, one_training_phase
from model_management.yuv import read_video, yuv_dict_to_device
from models.cool_chic import CoolChicParameter, to_device
from utils.device import get_best_device

"""
Use this file to encode an image.


Syntax example for the synthesis:
    12-1-linear,12-1-residual,3-1-linear,3-3-residual

    This is a 4 layers synthesis. Now the output layer (computing the final RGB
values) must be specified i.e. a 12,12 should now be called a 12,12,3. Each layer
is described using the following syntax:

        <output_dim>-<kernel_size>-<type>-<non_linearity>

    <output_dim>    : Number of output features.
    <kernel_size>   : Spatial dimension of the kernel. Use 1 to mimic an MLP.
    <type>          :   - "linear" for a standard conv,
                        - "residual" for a residual block i.e. layer(x) = x + conv(x)
                        - "attention" for an attention block i.e.
                            layer(x) = x + conv_1(x) * sigmoid(conv_2(x))
    <non_linearity> :   - "none" for no non-linearity,
                        - "relu" for a ReLU,
                        - "leakyrelu" for a LeakyReLU,
"""

if __name__ == '__main__':
    # =========================== Parse arguments =========================== #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device', type=str, default='auto',
        help='"auto" to find best available device otherwise "cpu" or "cuda:0" or "mps:0"'
    )
    # I/O paths ------------------------------------------------------------- #
    parser.add_argument('-i', '--input', help='Path of the input image.', type=str)
    parser.add_argument('-o', '--output', type=str, default='./bitstream.bin', help='Bitstream path.')
    parser.add_argument('--model_save_path', type=str, default='./model.pt', help='Save pytorch model here.')
    parser.add_argument('--enc_results_path', type=str, default='./encoder_results.txt', help='Save encoder-side results here.')
    # Encoding parameters --------------------------------------------------- #
    parser.add_argument('--lmbda', help='Rate constraint', type=float, default=1e-3)
    parser.add_argument('--start_lr', help='Initial learning rate', type=float, default=1e-2)
    parser.add_argument('--n_itr', help='Number of maximum iterations', type=int, default=int(1e6))
    parser.add_argument('--recipe', help='recipe type', type=str, default='slow')
    # Architecture ---------------------------------------------------------- #
    parser.add_argument(
        '--layers_synthesis', help='See default.', type=str,
        default='40-1-linear-relu,3-1-linear-relu,3-3-residual-relu,3-3-residual-none'
    )
    parser.add_argument('--layers_arm', help='Format: 16,8,16', type=str, default='24,24')
    parser.add_argument('--n_ctx_rowcol', help='Number of rows/columns for ARM', type=int, default=2)
    parser.add_argument('--latent_n_grids', help='Number of latent grids', type=int, default=7)
    parser.add_argument('--upsampling_kernel_size', help='upsampling kernel size (≥8)', type=int, default=8)
    args = parser.parse_args()
    # =========================== Parse arguments =========================== #

    # =========================== Parse arguments =========================== #
    layers_synthesis = [x for x in args.layers_synthesis.split(',') if x != '']
    layers_arm = [int(x) for x in args.layers_arm.split(',') if x != '']

    # Parse arguments
    layers_synthesis = [x for x in args.layers_synthesis.split(',') if x != '']
    layers_arm = [int(x) for x in args.layers_arm.split(',') if x != '']
    if args.device == 'auto':
        device = get_best_device()
    else:
        device = args.device
    print(f'Running on {device}')

    if args.input.endswith('.yuv'):
        # For now, we always read the first frame
        img = yuv_dict_to_device(read_video(args.input, frame_idx=0), device)
        img_type = '_'.join(args.input.split('/')[-1].split('.')[0].split('_')[-2:])
    else:
        img = to_tensor(Image.open(args.input)).unsqueeze(0).to(device)
        img_type = 'rgb444'

    # Create a CoolChicParameter object to store all these parameters
    cool_chic_param = CoolChicParameter(
        lmbda=args.lmbda,
        dist='mse',
        img=img,
        img_type=img_type,
        layers_synthesis=layers_synthesis,
        layers_arm=layers_arm,
        n_ctx_rowcol=args.n_ctx_rowcol,
        latent_n_grids=args.latent_n_grids,
        upsampling_kernel_size = args.upsampling_kernel_size,
        ste_derivative=0.01,
    )

    print(f'Image size: {cool_chic_param.img_size}')
    print(f'Image type: {cool_chic_param.img_type}')
    # =========================== Parse arguments =========================== #

    # ====================== Torchscript JIT parameters ===================== #
    # From https://github.com/pytorch/pytorch/issues/52286
    # This is no longer the case with the with torch.jit.fuser
    # ! This gives a significant (+25 %) speed up
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_profiling_mode(False)

    if device == 'cpu':
        # the number of cores is adjusted to the machine cpu count
        n_cores = int(os.cpu_count())
        print(f"Using {n_cores} cpu cores")

        torch.set_flush_denormal(True)
        torch.set_num_interop_threads(n_cores) # Inter-op parallelism
        torch.set_num_threads(n_cores) # Intra-op parallelism

        subprocess.call('export OMP_PROC_BIND=spread', shell=True)
        subprocess.call('export OMP_PLACES=threads', shell=True)
        subprocess.call('export OMP_SCHEDULE=static', shell=True)
        subprocess.call(f'export OMP_NUM_THREADS={n_cores}', shell=True)
        subprocess.call('export KMP_HW_SUBSET=1T', shell=True)

    elif device == 'mps:0':
        print(f'MS-SSIM computation does not work with YUV data on mps:0 device!')
    # ====================== Torchscript JIT parameters ===================== #


    # ===================== Initialize and train model ====================== #
    # Retrieve the preset class and instantiate it with the desired argument
    training_preset = AVAILABLE_PRESET.get(args.recipe)(start_lr=args.start_lr, dist='mse')
    if training_preset is None:
        print(f'Unknown training recipe. Found {args.recipe}')
        print(f'Expected: {AVAILABLE_PRESET.keys()}')
        print(f'Exiting')
        sys.exit(1)

    print(training_preset.to_string())

    # Perform warm-up to fine the best starting point
    model = do_warmup(cool_chic_param, training_preset, device=device)
    # model is sent back to CPU at the end of do_warmup
    model = to_device(model, device)

    # Do the different training phase
    for idx_phase, phase in enumerate(training_preset.all_phases):

        print(f'\n{"#" * 40}    Training phase: {idx_phase:>2}    {"#" * 40}\n')

        # 2. Actual training ------------------------------------------------ #
        model, results = one_training_phase(model, phase)

        print(f'\nPerformance at the end of the phase:')
        print(results.to_string(mode='short', print_col_name=True))

    # Dump the results into a file and save the model
    save_model(model, args.model_save_path)
    with open(args.enc_results_path, 'w') as f_out:
        f_out.write(results.to_string(mode='all', print_col_name=True) + '\n')

    # Print the final results
    print('\nFinal encoder-side results (quantized results):')
    print('-------------------------------------------------')
    print(results.to_string(mode='more', print_col_name=True))

    # ========================== Encode the network ========================= #
    print("Encoding to", args.output)
    encode(
        model,
        args.output,
        {
            'arm_weight':        results.q_step_arm_weight,
            'arm_bias':          results.q_step_arm_bias,
            'upsampling_weight': results.q_step_upsampling_weight,
            'upsampling_bias':   results.q_step_upsampling_bias,
            'synthesis_weight':  results.q_step_synthesis_weight,
            'synthesis_bias':    results.q_step_synthesis_bias,
        }
    )
    # ========================== Encode the network ========================= #
