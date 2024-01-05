# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import math
import typing
import torch
from torch import Tensor, nn
from typing import Any, List, Dict, Tuple, TypedDict
from dataclasses import dataclass, field, fields
from fvcore.nn import FlopCountAnalysis, flop_count_table

from models.coolchic_components.quantizer import NoiseQuantizer, STEQuantizer
from models.coolchic_components.synthesis import Synthesis
from models.coolchic_components.upsampling import Upsampling
from models.coolchic_components.arm import Arm, get_flat_latent_and_context, compute_rate
from utils.misc import POSSIBLE_DEVICE, ARMINT, DescriptorCoolChic


@dataclass
class CoolChicEncoderParameter():
    """Dataclass to store the parameters of CoolChicEncoder for one frame."""

    # ----- Data of the frame to encode
    # Height x Width of the biggest feature map and of the output.
    # Set by video_encoder
    img_size: Tuple[int, int] = field(init=False)

    # ----- Architecture options
    layers_synthesis: List[str]         # Synthesis architecture (e.g. '12-1-linear-relu', '12-1-residual-relu', '3-1-linear-relu', '3-3-residual-none')
    layers_arm: List[int]               # Output dim. of each hidden layer for the ARM (Empty for linear MLP)
    n_ctx_rowcol: int = 2               # Number of row and columns of context.
    upsampling_kernel_size: int = 8     # Kernel size for the upsampler ≥8. if set to zero the kernel is not optimised, the bicubic upsampler is used
    n_ft_per_res: List[int] = 1         # Number of features for each resolution.

    # ==================== Not set by the init function ===================== #
    latent_n_grids: int = field(init=False) # Number of different resolutions
    # ==================== Not set by the init function ===================== #


    def __post_init__(self):
        self.latent_n_grids = len(self.n_ft_per_res)

    def pretty_string(self) -> str:
        """Return a pretty string formatting the data within the class"""
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80

        s = 'CoolChicEncoderParameter value:\n'
        s += '-------------------------------\n'
        for k in fields(self):
            s += f'{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n'
        s += '\n'
        return s


class CoolChicEncoderOutput(TypedDict):
    """Dataclass representing the output of CoolChicEncoder forward."""
    raw_out: Tensor                     # Output of the synthesis forward [B, C, H, W]
    rate: Tensor                        # Rate associated to each latent [total_latent_value]

    # Any other data required to compute some logs, stored inside a dictionary
    additional_data: Dict[str, Any]

class CoolChicEncoder(nn.Module):
    """CoolChicEncoder for a single frame."""

    def __init__(self, param: CoolChicEncoderParameter):
        """Instantiate a cool-chic encoder for one frame.

        Args:
            param (CoolChicEncoderParameter): See above for more details
        """
        super().__init__()

        # ? What is this for?
        torch.set_printoptions(threshold=10000000)

        # Everything is stored inside param
        self.param = param

        # ================== Synthesis related stuff ================= #
        # Encoder-side latent gain applied prior to quantization, one per feature
        self.encoder_gains = torch.ones(self.param.latent_n_grids,) * 16

        # Populate the successive grids
        self.latent_grids = nn.ParameterList()
        dim_synthesis_input = 0
        for i in range(self.param.latent_n_grids):
            h_grid, w_grid = [int(math.ceil(x / (2 ** i))) for x in self.param.img_size]

            if isinstance(self.param.n_ft_per_res, list):
                c_grid = self.param.n_ft_per_res[i]
            else:
                c_grid = self.param.n_ft_per_res

            dim_synthesis_input += c_grid

            self.latent_grids.append(
                nn.Parameter(torch.zeros((1, c_grid, h_grid, w_grid)), requires_grad=True)
            )

        # Instantiate the synthesis MLP
        self.synthesis = Synthesis(dim_synthesis_input, self.param.layers_synthesis)
        # ================== Synthesis related stuff ================= #

        # ================ Quantization related stuff ================ #
        self.noise_quantizer = NoiseQuantizer()
        self.ste_quantizer = STEQuantizer()
        # ================ Quantization related stuff ================ #

        # ===================== Upsampling stuff ===================== #
        self.upsampling = Upsampling(self.param.upsampling_kernel_size)
        # ===================== Upsampling stuff ===================== #

        # ===================== ARM related stuff ==================== #
        # Create the probability model for the main INR. It uses a spatial context
        # parameterized by the spatial context

        # If we need 3 rows & columns of context, we'll use a 7x7 mask as:
        #   1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1
        #   1 1 1 * 0 0 0
        #   0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0

        # Mask of size 2N + 1 when we have N rows & columns of context.
        self.mask_size = 2 * self.param.n_ctx_rowcol + 1

        # Number of non-zero pixels in the (2N + 1) x (2N + 1) mask. This is the actual
        # context size. Context size = (N ** 2 - 1) / 2
        self.non_zero_pixel_ctx = int((self.mask_size ** 2 - 1) / 2)

        # Index in the mask (i.e. 0, 1, ... context_size - 1). Used for faster implementation
        # of the get_neighbors function. This allows to use the index_select function, which
        # is significantly faster than usual indexing.
        self.non_zero_pixel_ctx_index = torch.arange(0, self.non_zero_pixel_ctx)

        self.arm = Arm(self.non_zero_pixel_ctx, self.param.layers_arm)
        # ===================== ARM related stuff ==================== #

        # ======================== Monitoring ======================== #
        # # Pretty string representing the decoder complexity
        # self.flops_str = ""
        # # Total number of multiplications to decode the image
        # self.total_flops = 0.
        self.get_flops()
        # ======================== Monitoring ======================== #

        # Something like ['arm', 'synthesis', 'upsampling']
        self.modules_to_send = [tmp.name for tmp in fields(DescriptorCoolChic)]

    def get_flops(self):
        # Count the number of floating point operations here. It must be done before
        # torch scripting the different modules.
        flops = FlopCountAnalysis(self, None)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)

        self.total_flops = flops.total()
        self.flops_str = flop_count_table(flops)
        del flops

    def get_network_rate(self) -> DescriptorCoolChic:
        """Return the rate (in bits) associated to the parameters (weights and biases)
        of the different modules

        Returns:
            DescriptorCoolChic: The rate (in bits) associated with the weights and biases of each module
        """
        rate_per_module: DescriptorCoolChic = {
            module_name: {'weight': 0., 'bias': 0.} for module_name in self.modules_to_send
        }

        for module_name in self.modules_to_send:
            rate_per_module[module_name] = getattr(self, module_name).measure_laplace_rate()

        return rate_per_module

    def get_network_quantization_step(self) -> DescriptorCoolChic:
        """Return the quantization step associated to the parameters (weights and biases)
        of the different modules

        Returns:
            DescriptorCoolChic: The quantization step associated with the weights and biases of each module
        """
        q_step_per_module: DescriptorCoolChic = {
            module_name: {'weight': 0., 'bias': 0.} for module_name in self.modules_to_send
        }

        for module_name in self.modules_to_send:
            q_step_per_module[module_name] = getattr(self, module_name).get_q_step()

        return q_step_per_module

    def str_complexity(self) -> str:
        """Return a string describing the number of MAC (**not mac per pixel**) and the
        number of parameters for the different modules of CoolChic

        Returns:
            str: A pretty string about CoolChic complexity.
        """

        if not self.flops_str:
            self.get_flops()

        msg_total_mac = '----------------------------------\n'
        msg_total_mac += f'Total MAC / decoded pixel: {self.get_total_mac_per_pixel():.1f}'
        msg_total_mac += '\n----------------------------------'


        return self.flops_str + '\n\n' + msg_total_mac

    def get_total_mac_per_pixel(self) -> float:
        """Count the number of Multiplication-ACcumulation (MAC) per decoded pixel
        for this model.

        Returns:
            float: number of floating point operation per decoded pixel.
        """

        if not self.flops_str:
            self.get_flops()

        n_pixels = self.param.img_size[-2] * self.param.img_size[-1]
        return self.total_flops / n_pixels

    def get_quantized_latent(self, use_ste_quant: bool=True, AC_MAX_VAL: int=-1) -> List[Tensor]:
        """Scale and quantize the latent to obtain the sent latent. AC_MAX_VAL can be used
        to clamp the latent if we actually want to perform entropy coding.

        Args:
            use_ste_quant (bool, optional): True to use the straight-through estimator for
                quantization. Defaults to True.
            AC_MAX_VAL (int, optional): If different from -1, clamp the value in
                [-AC_MAX_VAL ; AC_MAX_VAL + 1] to write the actual bitstream. Defaults to -1.

        Returns:
            List[Tensor]: List of [1, C, H', W'] latent variable with H' and W' depending
                on the particular resolution of each latent.
        """
        scaled_latent = [
            cur_latent * self.encoder_gains[i] for i, cur_latent in enumerate(self.latent_grids)
        ]

        if self.training:
            if use_ste_quant:
                sent_latent = [self.ste_quantizer(cur_latent) for cur_latent in scaled_latent]
            else:
                sent_latent = [self.noise_quantizer(cur_latent) for cur_latent in scaled_latent]
        else:
            sent_latent = [torch.round(cur_latent) for cur_latent in scaled_latent]

        # Clamp latent if we need to write a bitstream
        if AC_MAX_VAL != -1:
            sent_latent = [
                torch.clamp(cur_latent, -AC_MAX_VAL, AC_MAX_VAL + 1) for cur_latent in sent_latent
            ]

        return sent_latent

    def get_latent_rate_and_proba(self, sent_latent: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the probability model and the rate in bits associated to the sent latent.

            /!\ All the returns of this function (rate, mu, scale) are flattened to comply
                with the way the ARM MLP represents the data. For instance, the rate is
                represented as a [N] tensor where N is the number of total latent variables.

        Args:
            sent_latent (List[Tensor]): List of each [B, C, H', W'] sent latent.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: flat rate, flat mu and flat scale. All one dimensional
                tensor with <total_number_of_latents> elements.
        """

        # Flatten the latent to comply with MLP arm
        # flat_latent = [N, 1] tensor describing N latents
        # flat_context = [N, context_size] tensor describing the context of each latent

        # Extract the spatial content and the latent to code
        flat_latent, flat_context = get_flat_latent_and_context(
            sent_latent, self.mask_size, self.non_zero_pixel_ctx_index
        )

        # Feed the spatial context to the arm MLP and get mu and scale
        raw_proba_param = self.arm(flat_context)

        # Compute the rate (i.e. the entropy of flat latent knowing mu and scale)
        flat_rate_y, flat_mu, flat_scale = compute_rate(flat_latent, raw_proba_param)

        return flat_rate_y, flat_mu, flat_scale

    def forward(
        self,
        use_ste_quant: bool = False,
        AC_MAX_VAL: int = -1,
        flag_additional_outputs: bool = False,
    ) -> CoolChicEncoderOutput:
        """Perform CoolChicEncoder forward pass, to be used during the training.
        The main step are as follows:
            1. Scale and quantize the encoder-side latent to get the latent ready to be sent;
            2. Measure the rate with the auto-regressive module;
            3. Upsample and synthesize the latent to get the output.

        Args:
            use_ste_quant (bool, optional): If True use the true quantization in the forward
                with its gradient set to the one of the softround.
                Else, use the additive noise in the forward with its gradient set to 100.
                Defaults to False.
            AC_MAX_VAL (int, optional): If different from -1, clamp the value in
                [-AC_MAX_VAL ; AC_MAX_VAL + 1] to write the actual bitstream. Defaults to -1.
            flag_additional_outputs (bool, optional): True to fill CoolChicEncoderOutput['additional_data']
                with many different quantities which can be used to analyze Cool-chic behavior.
                Defaults to False.

        Returns:
            CoolChicEncoderOutput: Output of Cool-chic training forward pass.
        """

        # ! Order of the operations are important as these are asynchronous
        # ! CUDA operations. Some ordering are faster than other...
        if ARMINT:
            self.non_zero_pixel_ctx_index = self.non_zero_pixel_ctx_index.to(self.latent_grids[0].device)
            self.min_gain = self.min_gain.to(self.latent_grids[0].device)

        sent_latent = self.get_quantized_latent(use_ste_quant=use_ste_quant, AC_MAX_VAL=AC_MAX_VAL)

        # ----- Decoder is just that!
        flat_rate, flat_mu, flat_scale = self.get_latent_rate_and_proba(sent_latent)
        synthesis_input = self.upsampling(sent_latent)
        synthesis_output = self.synthesis(synthesis_input)
        # ----- Decoder is just that!

        additional_data = {}
        if flag_additional_outputs == True:
            # Prepare list to accommodate the visualisations
            additional_data['detailed_sent_latent'] = []
            additional_data['detailed_mu'] = []
            additional_data['detailed_scale'] = []
            additional_data['detailed_rate_bit'] = []
            additional_data['detailed_centered_latent'] = []

            # "Pointer" for the reading of the 1D scale, mu and rate
            cnt = 0
            # for i, _ in enumerate(filtered_latent):
            for index_latent_res, _ in enumerate(self.latent_grids):
                c_i, h_i, w_i = sent_latent[index_latent_res].size()[-3:]
                additional_data['detailed_sent_latent'].append(sent_latent[index_latent_res].view((1, c_i, h_i, w_i)))

                # Scale, mu and rate are 1D tensors where the N latent grids
                # are flattened together. As such we have to read the appropriate
                # number of values in this 1D vector to reconstruct the i-th grid in 2D
                mu_i, scale_i, rate_i = [
                    # Read h_i * w_i values starting from cnt
                    tmp[cnt: cnt + (c_i * h_i * w_i)].view((1, c_i, h_i, w_i))
                    for tmp in [flat_mu, flat_scale, flat_rate]
                ]

                cnt += c_i * h_i * w_i
                additional_data['detailed_mu'].append(mu_i)
                additional_data['detailed_scale'].append(scale_i)
                additional_data['detailed_rate_bit'].append(rate_i)
                additional_data['detailed_centered_latent'].append(additional_data['detailed_sent_latent'][-1] - mu_i)


        res: CoolChicEncoderOutput = {
            'raw_out': synthesis_output,
            'rate': flat_rate,
            'additional_data': additional_data
        }

        return res

    def to_device(self, device: POSSIBLE_DEVICE):
        """Push a model to a given device.

        Args:
            device (POSSIBLE_DEVICE): The device on which the model should run.
        """

        assert device in typing.get_args(POSSIBLE_DEVICE),\
            f'Unknown device {device}, should be in {typing.get_args(POSSIBLE_DEVICE)}'
        self = self.to(device)
        self.non_zero_pixel_ctx_index = self.non_zero_pixel_ctx_index.to(device)
        self.encoder_gains = self.encoder_gains.to(device)

        # Push integerized weights and biases of the mlp (resp qw and qb) to
        # the required device
        for idx_layer, layer in enumerate(self.arm.mlp):
            if hasattr(layer, 'qw'):
                if layer.qw is not None:
                    self.arm.mlp[idx_layer].qw = layer.qw.to(device)

            if hasattr(layer, 'qb'):
                if layer.qb is not None:
                    self.arm.mlp[idx_layer].qb = layer.qb.to(device)
