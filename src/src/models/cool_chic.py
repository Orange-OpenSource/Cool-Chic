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
import typing
from torch import Tensor, nn
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, field
from fvcore.nn import FlopCountAnalysis, flop_count_table

from model_management.yuv import DictTensorYUV, convert_444_to_420, yuv_dict_clamp, yuv_dict_to_device
from models.synthesis import NoiseQuantizer, STEQuantizer, Synthesis
from models.arm import Arm, get_flat_latent_and_context, compute_rate
from models.upsampling import Upsampling
from utils.device import POSSIBLE_DEVICE
from utils.constants import ARMINT
from utils.data_structure import DescriptorCoolChic


@dataclass
class CoolChicParameter():
    """Dataclass to store the parameters of CoolChic."""
    lmbda: float                        # Rate constraint. Loss = D + lambda R
    img: Union[Tensor, DictTensorYUV]   # [1, C, H, W] 4D tensor storing the image to code or 3 4-d tensor for yuv420 (in [0., 1.])
    img_type: str                       # Either 'rgb444' or 'yuv420_8b' or 'yuv420_10b
    layers_synthesis: List[str]         # Output dim. of layer for the synthesis (e.g. 12-1-linear-relu,12-1-residual-relu,3-1-linear-relu,3-3-residual-none)
    layers_arm: List[int]               # Output dim. of each hidden layer for the ARM (Empty for linear MLP)
    n_ctx_rowcol: int = 2               # Number of row and columns of context.
    latent_n_grids: int = 7             # Number of latent grids.
    dist: str = 'mse'                   # Either "mse" or "ms_ssim"
    upsampling_kernel_size: int=8       # kernel size for the upsampler ≥8. if set to zero the kernel is not optimised, the bicubic upsampler is used
    bitdepth: int = 8                   # Either 8 bits or 10 bits. Only for YUV file
    ste_derivative: float = 1e-2        # Derivative used for the actual quantization

    # ==================== Not set by the init function ===================== #
    img_size: Tuple[int, int] = field(init=False)       # Height, Width of the image to code
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        if self.img_type == 'rgb444':
            self.img_size = self.img.size()[-2:]
        elif 'yuv420' in self.img_type:
            self.img_size = self.img.get('y').size()[-2:]
            if '10b' in self.img_type:
                self.bitdepth = 10


class CoolChicEncoder(nn.Module):

    def __init__(self, param: CoolChicParameter):
        """Instantiate an INR for this images

        Args:
            img_size (tuple, optional): (height, width) of the image. Defaults to None.
            layers_synthesis (list, optional): Dimension of the output of *each* hidden layer.
                if empty, no hidden layers
            layers_arm (list, optional): Dimension of the output of *each* hidden layer for the
                ARM MLP. if empty, no hidden layers
            n_ctx_rowcol (int, optional): How many row and columns are used as context by the ARM.
            latent_n_grids (int, optional): Number of latent grids (and number of resolution).
            post_process_mode (str, optional): Either "on" or "off".
        """
        super().__init__()

        # Store useful values and initialize iterations and training time counter
        self.param = param
        self.iterations_counter = 0
        self.total_training_time_sec = 0.0

        # ================== Synthesis related stuff ================= #
        # Empty grids and associated gains.
        self.log_2_encoder_gains = nn.Parameter(
            torch.arange(0., self.param.latent_n_grids), requires_grad=True
        )
        self.min_gain = torch.tensor([1.0], requires_grad=False)

        # Populate the successive grid
        self.latent_grids = nn.ParameterList()
        for i in range(self.param.latent_n_grids):
            h_grid, w_grid = [int(math.ceil(x / (2 ** i))) for x in self.param.img_size]

            self.latent_grids.append(
                nn.Parameter(
                    torch.zeros((1, 1, h_grid, w_grid)), requires_grad=True
                )
            )

        # Instantiate the synthesis MLP
        self.synthesis = Synthesis(self.param.latent_n_grids, self.param.layers_synthesis)

        self.noise_quantizer = NoiseQuantizer()
        self.ste_quantizer = STEQuantizer()

        # Not a very elegant way of setting the derivative
        STEQuantizer.ste_derivative = self.param.ste_derivative
        # ================== Synthesis related stuff ================= #

        # ================== Upsampling related stuff ================ #
        self.upsampling = Upsampling(param.upsampling_kernel_size)
        # ================== Upsampling related stuff ================ #


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
        self.mask_size = 2 * self.param.n_ctx_rowcol + 1

        self.non_zero_pixel_ctx = int((self.mask_size ** 2 - 1) / 2)

        # 1D tensor containing the indices of the non zero context
        # pixels (i.e. floor(N ** 2 / 2) - 1). It looks like:
        #       [0, 1, ..., floor(N ** 2 / 2) - 1].
        # This allows to use the index_select function, which is significantly
        # faster than usual indexing.
        self.non_zero_pixel_ctx_index = torch.arange(0, self.non_zero_pixel_ctx)
        self.arm = Arm(self.non_zero_pixel_ctx, self.param.layers_arm)
        # ===================== ARM related stuff ==================== #

        #  prepare to count the number of floating point operations
        self.flops_str = None

        torch.set_printoptions(threshold=10000000)


    def get_flops(self):
        # Count the number of floating point operations here. It must be done before
        # torchscripting the different modules.
        flops = FlopCountAnalysis(self, None)
        flops.unsupported_ops_warnings(False)

        self.total_flops = flops.total()
        self.flops_str = flop_count_table(flops)
        del flops

    def get_network_rate(self) -> DescriptorCoolChic:
        """Return the rate associated to the parameters (weights and biases)
        of the different modules

        Returns:
            DescriptorCoolChic: The rate associated with the weights and biases of each module
        """
        module_to_send = {
            'arm': self.arm, 'synthesis': self.synthesis, 'upsampling': self.upsampling
        }

        rate_per_module: DescriptorCoolChic = {
            'arm': {'weight': 0., 'bias': 0.},
            'synthesis': {'weight': 0., 'bias': 0.},
            'upsampling': {'weight': 0., 'bias': 0.},
        }
        for name, mod in module_to_send.items():
            rate_per_module[name] = mod.measure_laplace_rate()

        return rate_per_module

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

    def get_latent_gain(self) -> Tensor:
        """Return the latent gains vector

        Returns:
            Tensor: The latent gains vector
        """
        return torch.max(2 ** self.log_2_encoder_gains, self.min_gain)

    def forward(
        self,
        visu: bool = False,
        specific_latent_grid: List[int] = [],
        use_ste_quant: bool = True,
        AC_MAX_VAL: int = -1
    ) -> Dict[str, Tensor]:
        """Perform Cool-chic forward pass.
            - Quantize the latent variable
            - Synthesize **all** the output pixels
            - Run the ARM on **all** the latent
            - Measure the rate of all the latent

        Args:
            visu (bool, optional): True to output more things in the
                results dictionary (latents, rate, mu, scale). Defaults to False.
            specific_latent_grid (List[int], optional):
                Pass [k, l, m] to synthesize the output only from the combination
                of the k-th, l-th and m-th feature maps. Ignored if the list
                is empty. Defaults to [].

        Returns:
            Dict[str, Tensor]: The forward output.
        """

        # ! Order of the operations are important as these are asynchronous
        # ! CUDA operations. Some ordering are faster than other...
        if ARMINT:
            self.non_zero_pixel_ctx_index = self.non_zero_pixel_ctx_index.to(self.latent_grids[0].device)
            self.min_gain = self.min_gain.to(self.latent_grids[0].device)
        latent_gains = self.get_latent_gain()

        scaled_latent = [
            cur_latent * latent_gains[i] for i, cur_latent in enumerate(self.latent_grids)
        ]


        if self.training:
            if use_ste_quant:
                sent_latent = [
                    self.ste_quantizer.apply(cur_latent) for cur_latent in scaled_latent
                ]
            else:
                sent_latent = [
                    self.noise_quantizer.apply(cur_latent) for cur_latent in scaled_latent
                ]

        else:
            sent_latent = [torch.round(cur_latent) for cur_latent in scaled_latent]

        # Clamp latent if we need to write a bitstream
        if AC_MAX_VAL != -1:
            sent_latent = [
                torch.clamp(cur_latent, -AC_MAX_VAL, AC_MAX_VAL + 1)
                for cur_latent in sent_latent
            ]

        # Extract the spatial content and the latent to code
        flat_latent, flat_context = get_flat_latent_and_context(
            sent_latent,
            self.mask_size,
            self.non_zero_pixel_ctx_index
        )

        # Upsample the latents
        synthesis_input = self.upsampling(sent_latent)

        # Feed the spatial context to the arm MLP and get mu and scale
        raw_proba_param = self.arm(flat_context)

        # Mask all but the desired channel(s)
        if specific_latent_grid:
            mask = torch.zeros_like(synthesis_input)
            for desired_idx in specific_latent_grid:
                mask[:, desired_idx] = 1
            synthesis_input = synthesis_input * mask

        # Compute the rate (i.e. the entropy of flat latent knowing mu and scale)
        rate_y, _, _ = compute_rate(flat_latent, raw_proba_param)

        # Reconstruct the output
        synthesis_output = self.synthesis(synthesis_input)

        if self.param.img_type == 'rgb444':
            # Simulate the quantization which occurs when saving on a 256 level PNG file
            if not self.training:
                synthesis_output = torch.round((2 ** 8 - 1) * synthesis_output) / (2 ** 8 - 1)
            x_hat = torch.clamp(synthesis_output, 0., 1.)

        elif 'yuv420' in self.param.img_type:
            # Simulate the quantization which occurs when saving on a 8 bits or 10 bits YUV file
            if not self.training:
                synthesis_output = torch.round((2 ** self.param.bitdepth - 1) * synthesis_output) / (2 ** self.param.bitdepth - 1)

            x_hat = yuv_dict_clamp(convert_444_to_420(synthesis_output), 0., 1.)

        out: Dict = {'x_hat': x_hat, 'rate_y': rate_y}

        if visu:
            rate_y, mu_y, scale_y = compute_rate(flat_latent, raw_proba_param)

            # Prepare list to accommodate the visualisations
            out['2d_y_latent'] = []
            out['2d_y_mu'] = []
            out['2d_y_scale'] = []
            out['2d_y_rate'] = []

            # "Pointer" for the reading of the 1D scale, mu and rate
            cnt = 0
            # for i, _ in enumerate(filtered_latent):
            for i, _ in enumerate(self.latent_grids):
                h_i, w_i = sent_latent[i].size()[-2:]
                out['2d_y_latent'].append(sent_latent[i].view((h_i, w_i)))

                # Scale, mu and rate are 1D tensors where the N latent grids
                # are flattened together. As such we have to read the appropriate
                # number of values in this 1D vector to reconstruct the i-th grid in 2D

                mu_i, scale_i, rate_i = [
                    # Read h_i * w_i values starting from cnt
                    tmp[cnt: cnt + (h_i * w_i)].view((h_i, w_i))
                    for tmp in [mu_y, scale_y, rate_y]
                ]

                cnt += h_i * w_i
                out['2d_y_mu'].append(mu_i)
                out['2d_y_scale'].append(scale_i)
                out['2d_y_rate'].append(rate_i)

        return out


def to_device(model: CoolChicEncoder, device: POSSIBLE_DEVICE) -> CoolChicEncoder:
    """Push a model to a given device.

    Args:
        model (CoolChicEncoder): The model to push.
        device (POSSIBLE_DEVICE): The device on which the model should run. ("cpu", "cuda:0", "mps:0")

    Returns:
        CoolChicEncoder: The model pushed on the required device
    """

    assert device in typing.get_args(POSSIBLE_DEVICE),\
        f'Unknown device {device}, should be in {typing.get_args(POSSIBLE_DEVICE)}'
    model = model.to(device)
    model.non_zero_pixel_ctx_index = model.non_zero_pixel_ctx_index.to(device)
    model.min_gain = model.min_gain.to(device)

    # Push integerized weights and biases of the mlp (resp qw and qb) to
    # the required device
    for idx_layer, layer in enumerate(model.arm.mlp):
        if hasattr(layer, 'qw'):
            if layer.qw is not None:
                model.arm.mlp[idx_layer].qw = layer.qw.to(device)

        if hasattr(layer, 'qb'):
            if layer.qb is not None:
                model.arm.mlp[idx_layer].qb = layer.qb.to(device)

    if model.param.img_type == 'rgb444':
        model.param.img = model.param.img.to(device)
    elif 'yuv420' in model.param.img_type:
        model.param.img = yuv_dict_to_device(model.param.img, device)

    return model
