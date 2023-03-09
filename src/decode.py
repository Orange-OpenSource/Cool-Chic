# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import argparse
import subprocess
import torch

from torchvision.transforms.functional import to_pil_image
from bitstream.decode import decode

if __name__ == '__main__':
    # =========================== Parse arguments =========================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path of the bitstream to decode.', type=str)
    parser.add_argument('-o', '--output', help='Path to save the decoded image.', type=str)
    parser.add_argument('--device', help='"cpu" or "cuda:0"', type=str, default='cuda:0')
    args = parser.parse_args()
    # =========================== Parse arguments =========================== #

    # ====================== Torchscript JIT parameters ===================== #
    # From https://github.com/pytorch/pytorch/issues/52286
    # This is no longer the case with the with torch.jit.fuser
    # ! This gives a small (-10 %) speed up
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_profiling_mode(False)
    # ====================== Torchscript JIT parameters ===================== #

    subprocess.call("export CUBLAS_WORKSPACE_CONFIG=:4096:8", shell=True)
    subprocess.call("export CUBLAS_WORKSPACE_CONFIG=:16:8", shell=True)
    x_hat = decode(args.input, device=args.device)
    to_pil_image(x_hat.cpu().squeeze(0)).save(args.output)
