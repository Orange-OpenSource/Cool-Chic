# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import subprocess
import torch
import numpy as np
from models.coolchic_encoder import CoolChicEncoder


@torch.no_grad()
def write_nn_behavior(model: CoolChicEncoder, save_dir: str):
    """Save different parameters of a model (upsampling filters,
    encoder gains, arm input template) into tsv files located
    inside save_dir

    Args:
        model (CoolChicEncoder): Model to analyze
        save_dir (str): Where should the parameters be saved
    """

    save_dir = save_dir.rstrip('/') + '/'
    subprocess.call(f'mkdir -p {save_dir}', shell=True)

    # Save ARM template ----------------------------------------------------- #
    weight = model.arm.mlp[0].weight.data.clone().cpu().detach().numpy()
    bias = model.arm.mlp[0].bias.data.clone().cpu().detach().numpy()

    np.savetxt(f'{save_dir}arm_template_weight.tsv', weight, delimiter='\t', fmt='%.10f')
    np.savetxt(f'{save_dir}arm_template_bias.tsv', bias, delimiter='\t', fmt='%.10f')

    # Save synthesis first layer -------------------------------------------- #
    weight = model.synthesis.layers[0].conv_layer.weight.data.clone().cpu().detach()
    bias = model.synthesis.layers[0].conv_layer.bias.data.clone().cpu().detach()
    # [Feature out, Feature in, k_size, k_size], k_size should be one -> [Ft out, Ft in]
    weight = weight.squeeze(-1).squeeze(-1)
    np.savetxt(f'{save_dir}synthesis_0_weight.tsv', weight.numpy(), delimiter='\t', fmt='%.10f')
    np.savetxt(f'{save_dir}synthesis_0_bias.tsv', bias.numpy(), delimiter='\t', fmt='%.10f')

    # Save upsampling filter ------------------------------------------------ #
    weight = model.upsampling.upsampling_layer.weight.clone().cpu()
    weight = weight.squeeze(0).squeeze(0).detach().numpy()
    np.savetxt(f'{save_dir}upsampler.tsv', weight, delimiter='\t', fmt='%.10f')

    # Save the latent gains ------------------------------------------------- #
    gains = model.get_latent_gain().data.cpu().detach().numpy()
    np.savetxt(f'{save_dir}gain.tsv', gains, delimiter='\t', fmt='%.10f')




# @torch.no_grad()
# def write_arm_input_template(model: CoolChicEncoder, save_dir: str):
#     """Save the first layer weight and bias of the ARM into two
#     text files located in save_dir

#     Args:
#         model (CoolChicEncoder): A trained model to analyze
#         save_dir (str): Where should the parameters be saved
#     """
#     save_dir = save_dir.rstrip('/') + '/'
#     subprocess.call(f'mkdir -p {save_dir}', shell=True)

#     weight = model.arm.mlp[0].w.data.clone().cpu().detach().numpy()
#     bias = model.arm.mlp[0].b.data.clone().cpu().detach().numpy()
#     np.savetxt(f'{save_dir}weight.tsv', weight, delimiter='\t', fmt='%.10f')
#     np.savetxt(f'{save_dir}bias.tsv', bias, delimiter='\t', fmt='%.10f')


# @torch.no_grad()
# def write_upsampling_filter(model: CoolChicEncoder, save_path: str):
#     """Save the upsampling impulse responses of a CoolChicEncoder into
#     save_path.

#     Args:
#         model (CoolChicEncoder): Model to visualize.
#         save_path (str): Where to save the upsampling impulse response.
#     """
#     # save the 2D basis for visualization purpose
#     weight = model.upsampling.upsampling_layer.weight.clone().cpu()
#     weight = torch.squeeze(weight,dim=0)
#     weight = torch.squeeze(weight,dim=0)
#     weight = weight.detach().numpy()
#     np.savetxt(save_path, weight, delimiter='\t', fmt='%.10f')


# @torch.no_grad()
# def write_latent_gain(model: CoolChicEncoder, save_path: str):
#     """Save a plot showing the value of the latent gains.

#     Args:
#         model (CoolChicEncoder): Model to analyze
#         save_path (str): Where to save the latent gains graph
#     """
#     gains = model.get_latent_gain().data.cpu().detach().numpy()
#     np.savetxt(save_path, gains, delimiter='\t', fmt='%.10f')

#     # plt.stem(gains)
#     # plt.ylim(0, 100)
#     # plt.xlabel('Feature map index (0: highest resolution)')
#     # plt.ylabel('Encoder gain value')
#     # plt.title('The encoder gain value for different features')
#     # plt.tight_layout()
#     # plt.savefig(save_path)
