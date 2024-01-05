# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import argparse
import subprocess
import torch

from bitstream.decode import decode_video

if __name__ == '__main__':
    # =========================== Parse arguments =========================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path of the bitstream to decode.', type=str)
    parser.add_argument('-o', '--output', help='Path to save the decoded image.', type=str)
    args = parser.parse_args()
    # =========================== Parse arguments =========================== #

    device = 'cpu'
    if device == 'cpu':
        # the number of cores is adjusted to the machine cpu count
        n_cores = 2 # int(os.cpu_count())
        print(f"Using {n_cores} cpu cores")

        torch.set_flush_denormal(True)
        torch.set_num_threads(n_cores) # Intra-op parallelism
        torch.set_num_interop_threads(n_cores) # Inter-op parallelism

        subprocess.call('export OMP_PROC_BIND=spread', shell=True)
        subprocess.call('export OMP_PLACES=threads', shell=True)
        subprocess.call('export OMP_SCHEDULE=static', shell=True)
        subprocess.call(f'export OMP_NUM_THREADS={n_cores}', shell=True)
        subprocess.call('export KMP_HW_SUBSET=1T', shell=True)


    subprocess.call("export CUBLAS_WORKSPACE_CONFIG=:4096:8", shell=True)
    subprocess.call("export CUBLAS_WORKSPACE_CONFIG=:16:8", shell=True)
    decode_video(args.input, args.output)
