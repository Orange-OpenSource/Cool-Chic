# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import math
import torch
from torch import Tensor, nn
from typing import List, Dict, Tuple, TypedDict, Optional
from models.synthesis import STEQuantizer, SynthesisMLP, UniformNoiseQuantizer, get_synthesis_input_latent
from models.arm import ArmMLP, get_flat_latent_and_context, compute_rate
from utils.constants import LIST_POSSIBLE_DEVICES


class EncoderOutput(TypedDict):
    """Define the dictionary containing COOL-CHIC encoder output as a type."""
    x_hat: Tensor                      # Reconstructed frame [1, C, H, W]
    rate_y: Tensor                     # Rate [1]
    mu: Optional[List[Tensor]]         # List of N [H_i, W_i] tensors, mu for each latent grid (can be None)
    scale: Optional[List[Tensor]]      # List of N [H_i, W_i] tensors, scale for each latent grid (can be None)
    latent: Optional[List[Tensor]]     # List of N [H_i, W_i] tensors, each latent grid (can be None)


class CoolChicEncoder(nn.Module):

    def __init__(
        self,
        img_size: Tuple[int, int],
        layers_synthesis: List = [32, 32, 32],
        layers_arm: List = [16, 16, 16, 16],
        n_ctx_rowcol: int = 3,
        latent_n_grids: int = 7,
    ):
        """Instantiate an INR for this images

        Args:
            img_size (tuple, optional): (height, width) of the image. Defaults to None.
            layers_synthesis (list, optional): Dimension of the output of *each* hidden layer.
                if empty, no hidden layers
            layers_arm (list, optional): Dimension of the output of *each* hidden layer for the
                ARM MLP. if empty, no hidden layers
            n_ctx_rowcol (int, optional): How many row and columns are used as context by the ARM.
            latent_n_grids (int, optional): Number of latent grids (and number of resolution).
        """
        super().__init__()

        # Store useful value
        self.img_size: Tuple[int, int] = img_size
        self.n_ctx_rowcol = n_ctx_rowcol
        self.layers_arm = layers_arm
        self.layers_synthesis = layers_synthesis
        self.latent_n_grids = latent_n_grids

        # Needed at one point during the training
        self.ste_quantizer = STEQuantizer()
        self.uniform_noise_quantizer = UniformNoiseQuantizer()

        # ================== Synthesis related stuff ================= #
        # Empty grids and associated encoder gains.
        self.log_2_encoder_gains = nn.Parameter(
            torch.arange(0., latent_n_grids), requires_grad=True
        )

        # Populate the successive grids
        self.latent_grids = nn.ParameterList()
        for i in range(latent_n_grids):
            h_grid, w_grid = [int(math.ceil(x / (2 ** i))) for x in img_size]

            self.latent_grids.append(
                nn.Parameter(
                    torch.zeros((1, 1, h_grid, w_grid)), requires_grad=True
                )
            )

        # Instantiate the synthesis MLP
        self.synthesis = torch.jit.script(SynthesisMLP(latent_n_grids, layers_synthesis))
        # ================== Synthesis related stuff ================= #

        # ===================== ARM related stuff ==================== #
        # Create the probability model for the main INR. It uses a spatial context
        # parametered by the spatial context

        # If we need 3 rows & columns of context, we'll use a 7x7 mask as:
        #   1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1
        #   1 1 1 * 0 0 0
        #   0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0
        self.mask_size = 2 * n_ctx_rowcol + 1

        self.non_zero_pixel_ctx = int((self.mask_size ** 2 - 1) / 2)

        # 1D tensor containing the indices of the non zero context
        # pixels (i.e. floor(N ** 2 / 2) - 1). It looks like:
        #       [0, 1, ..., floor(N ** 2 / 2) - 1].
        # This allows to use the index_select function, which is significantly
        # faster than usual indexing.
        self.non_zero_pixel_ctx_index = torch.arange(0, self.non_zero_pixel_ctx)
        self.arm = torch.jit.script(ArmMLP(self.non_zero_pixel_ctx, layers_arm))
        # ===================== ARM related stuff ==================== #

    def get_nb_mac(self) -> Dict[str, float]:
        """Count the number of Multiplication-ACcumulation (MAC) per pixel for
        this model. Return a dictionary containing the MAC for the synthesis,
        the probability model and the overall system.
        This is an estimate which omits the upscaling, the quantization gain
        and the Laplace computation.

        Returns:
            dict: {
                'arm_mac_per_pixel': xxx,
                'synth_mac_per_pixel': yyy,
                'total_mal_per_pixel': zzz
            }
        """
        n_pixels = self.img_size[-2] * self.img_size[-1]

        # ========================= Compute MAC ARM ========================= #
        n_pixel_latent = self.get_nb_parameters()['inr_latent_grid']
        # Successive dimension of the ARM ARM:
        # [in_ft, hidden_layer_1, ..., last_hidden_layer, 2]
        dim_arm = [self.non_zero_pixel_ctx] + self.layers_arm + [2]
        n_mac_per_latent_pixel_arm = sum(
            [in_ft * out_ft for in_ft, out_ft in zip(dim_arm[:-1], dim_arm[1:])]
        )
        n_mac_per_pixel_arm = (n_pixel_latent / n_pixels) * n_mac_per_latent_pixel_arm
        # ========================= Compute MAC ARM ========================= #

        # ====================== Compute MAC Synthesis ======================= #
        # Successive dimension of the Synthesis MLP:
        # [in_ft, hidden_layer_1, ..., last_hidden_layer, 3]
        dim_synth = [self.latent_n_grids] + self.layers_synthesis + [3]
        n_mac_per_pixel_synth = sum(
            [in_ft * out_ft for in_ft, out_ft in zip(dim_synth[:-1], dim_synth[1:])]
        )
        # ====================== Compute MAC Synthesis ======================= #

        return {
            'arm_mac_per_pixel': n_mac_per_pixel_arm,
            'synth_mac_per_pixel': n_mac_per_pixel_synth,
            'total_mac_per_pixel': n_mac_per_pixel_synth + n_mac_per_pixel_arm,
        }

    def get_nb_parameters(self) -> dict:
        """Return the number of parameters of different system parts.

        Returns:
            dict: contains the number of parameters."""

        # return {k: v for k, v in self.inr.get_nb_parameters().items()}
        return {
            'inr_latent_grid': sum(
                p.numel() for p in self.latent_grids.parameters() if p.requires_grad
            ),
            'inr_mlp': sum(
                p.numel() for p in self.synthesis.parameters() if p.requires_grad
            ),
            'inr_proba_model': sum(
                p.numel() for p in self.arm.parameters() if p.requires_grad
            ),
        }

    def print_nb_mac(self) -> str:
        """Return a string describing the number of MAC/pixel in this INR.

        Returns:
            str: a pretty and informative string"""
        n_macs = self.get_nb_mac()

        s = '-' * 80 + '\n'
        s += f'Synthesis MLP kMAC / pixel: {n_macs.get("synth_mac_per_pixel") / 1000:6.3}\n'
        s += f'ARM MLP       kMAC / pixel: {n_macs.get("arm_mac_per_pixel") / 1000:6.3}\n'
        s += f'Total         kMAC / pixel: {n_macs.get("total_mac_per_pixel") / 1000:6.3}\n'
        s += '-' * 80 + '\n'
        return s

    def print_nb_parameters(self) -> str:
        """Return a string describing the number of parameters in this INR.

        Returns:
            str: a pretty and informative string"""
        n_params = self.get_nb_parameters()

        s = '-' * 80 + '\n'
        s += f'Latent        kParameter  : {n_params.get("inr_latent_grid") / 1000:7.4}\n'
        s += f'Synthesis MLP kParameter  : {n_params.get("inr_mlp") / 1000:7.4}\n'
        s += f'ARM MLP       kParameter  : {n_params.get("inr_proba_model") / 1000:7.4}\n'
        s += '-' * 80 + '\n'
        return s

    def forward(
        self,
        get_proba_param: bool = False,
        use_ste_quant: bool = True,
        AC_MAX_VAL: int = -1
    ) -> EncoderOutput:
        """Perform Cool-chic forward pass.
            - Quantize the latent variable
            - Synthesize **all** the output pixels
            - Run the ARM on **all** the latent
            - Measure the rate of all the latent

        Args:
            get_proba_param (bool, optional): True to also return mu and scale.
                This is needed for the bitstream. Defaults to False.
            specific_latent_grid (List[int], optional):
                Pass [k, l, m] to synthesize the output only from the combination
                of the k-th, l-th and m-th feature maps. Ignored if the list
                is empty. Defaults to [].

        Returns:
            EncoderOutput: The forward output.
        """

        # ! Order of the operations are important as these are asynchronous
        # ! CUDA operations. Some ordering are faster than other...

        # ====================== Get sent latent codes ====================== #
        # Two different types of quantization. quantize() function uses the usual
        # noise addition proxy if self.training is True and the actual round
        # otherwise.
        # if use_ste_quant the straight-through estimator is used i.e. actual
        # quantization in the forward pass and gradient set to one in the backward.
        quantizer = self.ste_quantizer if use_ste_quant else self.uniform_noise_quantizer
        sent_latent = [
            quantizer.apply(
                cur_latent * torch.pow(2, self.log_2_encoder_gains[i]), # Apply Q. step
                self.training                                           # Noise if needed
            )
            for i, cur_latent in enumerate(self.latent_grids)
        ]

        # Clamp latent if we need to write a bitstream
        if AC_MAX_VAL != -1:
            sent_latent = [
                torch.clamp(cur_latent, -AC_MAX_VAL, AC_MAX_VAL + 1)
                for cur_latent in sent_latent
            ]
        # ====================== Get sent latent codes ====================== #

        # Extract the spatial content and the latent to code
        flat_latent, flat_context = get_flat_latent_and_context(
            sent_latent, self.mask_size, self.non_zero_pixel_ctx_index
        )

        synthesis_input = get_synthesis_input_latent(sent_latent)

        # Feed the spatial context to the arm MLP and get mu and scale
        raw_proba_param = self.arm(flat_context)

        # Compute the rate (i.e. the entropy of flat latent knowing mu and scale)
        rate_y, _, _ = compute_rate(flat_latent, raw_proba_param)

        # Reconstruct the output
        x_hat = torch.clamp(self.synthesis(synthesis_input), 0., 1.)


        if get_proba_param:
            _, flat_mu_y, flat_scale_y = compute_rate(flat_latent, raw_proba_param)

            # Prepare list to accommodate the visualisations
            mu = []
            scale = []
            latent = []

            # "Pointer" for the reading of the 1D scale, mu and rate
            cnt = 0
            for i, _ in enumerate(self.latent_grids):
                h_i, w_i = sent_latent[i].size()[-2:]

                # Scale, mu and rate are 1D tensors where the N latent grids
                # are flattened together. As such we have to read the appropriate
                # number of values in this 1D vector to reconstruct the i-th grid in 2D
                mu_i, scale_i = [
                    # Read h_i * w_i values starting from cnt
                    tmp[cnt: cnt + (h_i * w_i)].view((h_i, w_i))
                    for tmp in [flat_mu_y, flat_scale_y]
                ]

                cnt += h_i * w_i
                mu.append(mu_i)
                scale.append(scale_i)
                latent.append(sent_latent[i].view(h_i, w_i))

        else:
            mu = None
            scale = None
            latent = None

        out: EncoderOutput = {
            'x_hat': x_hat,
            'rate_y': rate_y.sum(),
            'mu': mu,
            'scale': scale,
            'latent': latent,
        }
        return out


def to_device(model: CoolChicEncoder, device: str) -> CoolChicEncoder:
    """Push a model to a given device.

    Args:
        model (CoolChicEncoder): The model to push.
        device (str): The device on which the model should run. ("cpu", "cuda:0")

    Returns:
        CoolChicEncoder: The model pushed on the required device
    """

    assert device in LIST_POSSIBLE_DEVICES, f'Unknown device {device}, should be in {LIST_POSSIBLE_DEVICES}'
    model = model.to(device)
    model.non_zero_pixel_ctx_index = model.non_zero_pixel_ctx_index.to(device)
    return model