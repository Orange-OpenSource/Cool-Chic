# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
import subprocess
import argparse

from torch import Tensor
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from bitstream.encode import encode
from bitstream.decode import decode

from models.mlp_coding import greedy_quantization
from models.cool_chic import CoolChicEncoder, to_device

from utils.logging import format_results_loss
from model_management.trainer import loss_fn, train, save_model

if __name__ == '__main__':
    # =========================== Parse arguments =========================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='"cpu" or "cuda:0"', type=str, default='cuda:0')
    parser.add_argument('-i', '--input', help='Path of the input image.', type=str)
    parser.add_argument('-o', '--output', type=str, default='./bitstream.bin', help='Bitstream path.')
    parser.add_argument('--decoded_img_path', type=str, default='./decoded.png', help='Decoded image path.')
    parser.add_argument('--model_save_path', type=str, default='./model.pt', help='Save pytorch model here.')
    parser.add_argument('--enc_results_path', type=str, default='./encoder_results.txt', help='Save encoder-side results here.')
    parser.add_argument('--lmbda', help='Rate constraint', type=float, default=1e-3)
    parser.add_argument('--start_lr', help='Initial learning rate', type=float, default=1e-2)
    parser.add_argument('--n_itr', help='Number of maximum iterations', type=int, default=int(1e6))
    parser.add_argument('--layers_synthesis', help='Format: 16,8,16', type=str, default='12,12')
    parser.add_argument('--layers_arm', help='Format: 16,8,16', type=str, default='12,12')
    parser.add_argument('--n_ctx_rowcol', help='Number of rows/columns for ARM', type=int, default=2)
    parser.add_argument('--latent_n_grids', help='Number of latent grids', type=int, default=7)
    parser.add_argument('--n_trial_warmup', help='Number of warm-up trials', type=int, default=5)
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
    layers_synthesis = [int(x) for x in args.layers_synthesis.split(',') if x != '']
    layers_arm = [int(x) for x in args.layers_arm.split(',') if x != '']
    # =========================== Parse arguments =========================== #

    # ===================== Initialize and train model ====================== #
    img = to_tensor(Image.open(args.input)).unsqueeze(0).to(args.device)
    print(f'Image size: {img.size()}')

    # Do some loops with few iterations to find the best starting point
    best_loss = 1e6
    N_WARM_UP = args.n_trial_warmup
    for n_trial in range(N_WARM_UP):
        model = to_device(
            CoolChicEncoder(
                img.size()[-2:],
                layers_synthesis=layers_synthesis,
                layers_arm=layers_arm,
                n_ctx_rowcol=args.n_ctx_rowcol,
                latent_n_grids=args.latent_n_grids,
            ),
            args.device
        )

        cur_model, training_stat_logs = train(
            model, img, lmbda=args.lmbda, start_lr=args.start_lr, n_itr=200,
        )
        cur_loss = training_stat_logs['loss']

        if cur_loss < best_loss:
            best_loss = cur_loss
            start_model = cur_model

        print(f'Warm-up trial {n_trial + 1: >3} / {N_WARM_UP} ; loss = {1000 * cur_loss:4.3f} ; best_loss = {1000 * best_loss:4.3f}')

        model = start_model

    print(model)
    print(model.print_nb_parameters())
    print(model.print_nb_mac())

    model, training_stat_logs = train(
        model, img, lmbda=args.lmbda, start_lr=args.start_lr, n_itr=int(args.n_itr)
    )

    # Save non quantized model at the end of the training
    save_model(model, args.model_save_path)
    # ===================== Initialize and train model ====================== #

    # ========= Print full precision performance as a sanity check ========== #
    with torch.no_grad():
        model = model.eval()
        model_out = model()

        # Compute results
        _, metrics = loss_fn(model_out, img, args.lmbda, compute_logs=True)

        loss_float = metrics.get('loss').item()
        print('=' * 120)
        print('Full precision results')
        print(format_results_loss(metrics, col_name=True))
        print('=' * 120)
    # ========= Print full precision performance as a sanity check ========== #

    # ======================== Quantize the network ========================= #
    model, q_step, rate_mlp = greedy_quantization(model, img, args.lmbda, verbose=False)
    # Rate in bit, not in bpp
    total_mlp_rate = sum(v for _, v in rate_mlp.items())
    n_pixels = img.size()[-2] * img.size()[-1]
    print(f'Rate ARM        : {rate_mlp.get("arm") / n_pixels: 5.4f} bpp')
    print(f'Rate Synthesis  : {rate_mlp.get("synthesis") / n_pixels:5.4f} bpp')
    # ======================== Quantize the network ========================= #

    # ================ Final forward to check the performance =============== #
    with torch.no_grad():
        model = model.eval()
        model_out = model()

        # Compute results
        _, metrics = loss_fn(model_out, img, args.lmbda, compute_logs=True, rate_mlp=total_mlp_rate)
        loss_quantized = metrics.get('loss').item()

    print('=' * 120)
    print('Final (quantized) results')
    print(format_results_loss(metrics, col_name=True))
    print('=' * 120)
    # ================ Final forward to check the performance =============== #

    # ======================= Log results into a file ======================= #
    # Rate already in bpp in metrics
    rate = {}
    rate['rate_latent_bpp'] = metrics.get('rate_y')
    rate['rate_mlp_bpp'] = metrics.get('rate_mlp')
    rate['rate_all_bpp'] = sum(metrics.get(k) for k in ['rate_y', 'rate_mlp'])

    str_keys, str_vals = '', ''
    for k in ['psnr']:
        str_keys += f'{k},'
        str_vals += f'{metrics.get(k):8.6f},'

    for k, v in rate.items():
        if isinstance(v, Tensor):
            v = v.detach().item()
        str_keys += f'{k},'
        str_vals += f'{v},'

    for k, v in training_stat_logs.items():
        if k == 'loss':
            continue
        str_keys += f'{k},'
        str_vals += f'{v:7.1f},'

    # Remove coma at the end
    str_keys = str_keys.rstrip(',')
    str_vals = str_vals.rstrip(',')

    with open(args.enc_results_path, 'w') as f:
        f.write(str_keys + '\n' + str_vals + '\n')
    # ======================= Log results into a file ======================= #

    # ======= Store the overfitted weights and latent into a bitstream ====== #
    subprocess.call("export CUBLAS_WORKSPACE_CONFIG=:4096:8", shell=True)
    subprocess.call("export CUBLAS_WORKSPACE_CONFIG=:16:8", shell=True)
    real_rate_byte = encode(model, args.output, q_step_nn=q_step)
    real_rate_bpp = real_rate_byte * 8 / n_pixels
    # ======= Store the overfitted weights and latent into a bitstream ====== #

    # === Perform decoding at the encoder-side to measure the performance === #
    x_hat = decode(args.output, device=args.device)

    # Save decoded image
    to_pil_image(x_hat.cpu().squeeze(0)).save(args.decoded_img_path)

    # Measure PSNR
    real_psnr = -10 * torch.log10(((x_hat - img) ** 2).mean())
    print('\nDecoder-side performance:')
    print(f'rate_bpp\tpsnr_db')
    print(f'{real_rate_bpp:6.4f}\t{real_psnr: 7.4f}')

    # Append this to the encoder results file:

    # Re-read the existing file
    str_keys, str_vals = [x.rstrip('\n') for x in open(args.enc_results_path, 'r').readlines() if x]
    str_keys += f',decoder_rate_bpp,decoder_psnr_db'
    str_vals += f',{real_rate_bpp:7.5f},{real_psnr:8.5f}'

    # Re-write the existing file
    with open(args.enc_results_path, 'w') as f:
        f.write(str_keys + '\n' + str_vals + '\n')
    # === Perform decoding at the encoder-side to measure the performance === #
