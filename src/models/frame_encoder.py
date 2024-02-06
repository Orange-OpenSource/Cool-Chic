# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


"""A frame encoder is composed of a CoolChicEncoder and a InterCodingModule."""

import copy
import itertools
import math
import subprocess
import sys
import time
import typing
import torch

from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, OrderedDict, Tuple, Union
from torch import Tensor, nn
from dataclasses import dataclass, field, fields

from encoding_management.coding_structure import Frame, FrameData
from encoding_management.loss.loss import DistortionWeighting, compute_mse, compute_msssim
from encoding_management.presets import AVAILABLE_PRESETS, Preset, TrainerPhase
from utils.yuv import DictTensorYUV, convert_420_to_444, convert_444_to_420, load_frame_data_from_file, yuv_dict_clamp
from models.coolchic_encoder import CoolChicEncoder, CoolChicEncoderParameter
from models.inter_coding_module import InterCodingModule
from utils.misc import ARMINT, FIXED_POINT_FRACTIONAL_MULT, TrainingExitCode, is_job_over, DescriptorCoolChic, DescriptorNN, POSSIBLE_DEVICE
from visu.utils import save_visualisation


@dataclass
class FrameEncoderOutput():
    """Dataclass representing the output of FrameEncoder forward."""
    # Either a [B, 3, H, W] tensor representing the decoded image or a
    # dictionary with the following keys:
    #   {
    #         'y': [B, 1, H, W],
    #         'u': [B, 1, H / 2, W / 2],
    #         'v': [B, 1, H / 2, W / 2],
    #   }
    # Note: yuv444 data are represented as a simple [B, 3, H, W] tensor
    decoded_image: Union[Tensor, DictTensorYUV]
    rate: Tensor              # Rate associated to each latent [total_latent_value]

    # Any other data required to compute some logs, stored inside a dictionary
    additional_data: Dict[str, Any] = field(default_factory = lambda: {})

@dataclass
class FrameEncoderManager():
    """
    All the encoding option for a frame (loss, lambda, learning rate) as well as some
    counters monitoring the training time, the number of training iterations or the number
    of loops already done.
    """
    # ----- Encoding (i.e. training) options
    preset_name: str                                            # Preset name, should be a key in AVAILABLE_PRESETS src/encoding_management/presets.py
    dist_weight: DistortionWeighting                            # Distortion used during the encoding
    start_lr: float = 1e-2                                      # Initial learning rate
    lmbda: float = 1e-3                                         # Rate constraint. Loss = D + lmbda R
    n_loops: int = 1                                            # Number of training loop
    n_itr: int = int(1e5)                                       # Maximum number of training iterations for a **single** phase

    # ==================== Not set by the init function ===================== #
    # ----- Actual preset, instantiated from its name
    preset: Preset = field(init=False)                          # It contains the learning rate in the different phase

    # ----- Monitoring
    idx_best_loop: int = field(default=0, init=False)           # Index of the loop which gives the best results (i.e. the best_loss)
    best_loss: float = field(default=1e6, init=False)           # Overall best loss (for all loops)
    loop_counter: int = field(default=0, init=False)            # Number of loops already done
    loop_start_time: float = field(default=0., init=False)      # Loop start time (before warm-up) ? What's this?
    iterations_counter: int = field(default=0, init=False)      # Total number of iterations done, including warm-up
    total_training_time_sec: float = field(default=0.0, init=False) # Total training time (second), including warm-up
    phase_idx: int = field(default=0, init=False)               # Index of the current training phase for the current loop
    warm_up_done: bool = field(default=False, init=False)       # True if the warm-up has already been done for this loop
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        assert self.preset_name in AVAILABLE_PRESETS, f'Preset named {self.preset_name} does not exist.' \
            f' List of available preset:\n{list(AVAILABLE_PRESETS.keys())}.'

        self.preset = AVAILABLE_PRESETS.get(self.preset_name)(start_lr= self.start_lr, n_itr_per_phase=self.n_itr)

        flag_quantize_model = False
        for training_phase in self.preset.all_phases:
            if training_phase.quantize_model:
                flag_quantize_model = True
        assert flag_quantize_model, f'The selected preset ({self.preset_name}) does not include ' \
            f' a training phase with neural network quantization.\n{self.preset.pretty_string()}'


    def record_beaten(self, candidate_loss: float) -> bool:
        """Return True if the candidate loss is better (i.e. lower) than the best loss.

        Args:
            candidate_loss (float): Current candidate loss.

        Returns:
            bool: True if the candidate loss is better than the best loss
                (i.e. candidate < best).
        """
        return candidate_loss < self.best_loss

    def set_best_loss(self, new_best_loss: float):
        """Set the new best loss attribute. It automatically looks at the current loop_counter
        to fill the idx_best_loop attribute.

        Args:
            new_best_loss (float): The new best loss obtained at the current loop
        """
        self.best_loss = new_best_loss
        self.idx_best_loop = self.loop_counter


    def pretty_string(self) -> str:
        """Return a pretty string formatting the data within the class"""
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80

        s = 'FrameEncoderManager value:\n'
        s += '--------------------------\n'
        for k in fields(self):
            if k.name == 'preset':
                # Don't print preset, it's quite ugly
                continue

            s += f'{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n'
        s += '\n'
        return s


@dataclass(kw_only=True)
class LossFunctionOutput():
    """Output for FrameEncoder.loss_function"""
    # ----- This is the important output
    # Optional to allow easy inheritance by FrameEncoderLogs
    loss: Optional[float] = None                                        # The RD cost to optimize

    # Any other data required to compute some logs, stored inside a dictionary
    mse: Optional[float] = None                                         # Mean squared error                     [ / ]
    ms_ssim: Optional[float] = None                                     # Multi-scale similarity metric          [ / ]
    lpips: Optional[float] = None                                       # LPIPS score                            [ / ]
    rate_nn_bpp: Optional[float] = None                                 # Rate associated to the neural networks [bpp]
    rate_latent_bpp: Optional[float] = None                             # Rate associated to the latent          [bpp]

    # ==================== Not set by the init function ===================== #
    # Everything here is derived from the above metrics
    psnr_db: Optional[float] = field(init=False, default=None)          # PSNR                                  [ dB]
    ms_ssim_db: Optional[float] = field(init=False, default=None)       # MS-SSIM on a log scale                [ dB]
    lpips_db: Optional[float] = field(init=False, default=None)         # LPIPS on a log scale                  [ dB]
    total_rate_bpp: Optional[float] = field(init=False, default=None)   # Overall rate: latent & NNs            [bpp]
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        if self.mse is not None:
            self.psnr_db = -10. * math.log10(self.mse)

        if self.lpips is not None:
            self.lpips_db = -10. * math.log10(self.lpips)

        if self.ms_ssim is not None:
            self.ms_ssim_db = -10. * math.log10(1 - self.ms_ssim)

        if self.rate_nn_bpp is not None and self.rate_latent_bpp is not None:
            self.total_rate_bpp = self.rate_nn_bpp + self.rate_latent_bpp

@dataclass
class FrameEncoderLogs(LossFunctionOutput):
    """Output of the test function i.e. the actual results of the encoding
    of one frame by the frame encoder.

    It inherits from LossFunctionOutput, meaning that all attributes of LossFunctionOutput
    are also attributes of FrameEncoderLogs. A FrameEncoderLogs is thus initialized
    from a LossFunctionOutput, all attribute of the LossFunctionOutput  will be copied as
    new attributes for the class.

    This is what is going to be saved to a log file.
    """
    loss_function_output: LossFunctionOutput        # All outputs from the loss function, will be copied is __post_init__
    frame_encoder_output: FrameEncoderOutput        # Output of frame encoder forward
    original_frame: Frame                           # Non coded frame

    detailed_rate_nn: DescriptorCoolChic            # Rate for each NN weights & bias   [bit]
    quantization_param_nn: DescriptorCoolChic       # Quantization step for each NN weights & bias [ / ]

    lmbda: float                                    # Rate constraint in D + lambda * R [ / ]
    encoding_time_second: float                     # Duration of the encoding          [sec]
    encoding_iterations_cnt: int                    # Number of encoding iterations     [ / ]
    mac_decoded_pixel: float = 0.                       # Number of multiplication per decoded pixel

    # ==================== Not set by the init function ===================== #
    # Everything here is derived from frame_encoder_output and original_frame

    # ----- CoolChicEncoder outputs
    # Spatial distribution of the rate, obtained by summing the rate of the different features
    # for each spatial location (in bit). [1, 1, H, W]
    spatial_rate_bit: Optional[Tensor] = field(init=False)
    # Feature distribution of the rate, obtained by the summing all the spatial location
    # of a given feature. [Number of latent resolution]
    feature_rate_bpp: Optional[List[float]] = field(init=False, default_factory=lambda: [])

    # ----- Inter coding module outputs
    alpha: Optional[Tensor] = field(init=False, default=None)       # Inter / intra switch
    beta: Optional[Tensor] = field(init=False, default=None)        # Bi-directional prediction weighting
    residue: Optional[Tensor] = field(init=False, default=None)     # Residue
    flow_1: Optional[Tensor] = field(init=False, default=None)      # Optical flow for the first reference
    flow_2: Optional[Tensor] = field(init=False, default=None)      # Optical flow for the second reference
    prediction: Optional[Tensor] = field(init=False, default=None)  # Temporal prediction
    masked_prediction: Optional[Tensor] = field(init=False, default=None)   # Temporal prediction * alpha

    # ----- Compute prediction performance
    alpha_mean: Optional[float] = field(init=False, default=None)   # Mean value of alpha
    beta_mean: Optional[float] = field(init=False, default=None)    # Mean value of beta
    prediction_psnr_db: Optional[float] = field(init=False, default=None)   # PSNR of the prediction
    dummy_prediction_psnr_db: Optional[float] = field(init=False, default=None) # PSNR of a prediction if we had no motion

    # ----- Miscellaneous quantities recovered from self.frame
    img_size: Tuple[int, int] = field(init=False)                   # [Height, Width]
    n_pixels: int = field(init=False)                               # Height x Width
    display_order: int = field(init=False)                          # Index of the current frame in display order
    coding_order: int = field(init=False)                           # Index of the current frame in coding order
    seq_name: str = field(init=False)                               # Name of the sequence to which this frame belong

    # ----- Neural network rate in bit per pixels
    detailed_rate_nn_bpp: DescriptorCoolChic = field(init=False)    # Rate for each NN weights & bias   [bpp]

    def __post_init__(self):
        # ----- Copy all the attributes of loss_function_output
        for field in fields(self.loss_function_output):
            setattr(self, field.name, getattr(self.loss_function_output, field.name))

        # ----- Retrieve info from the frame
        self.img_size = self.original_frame.data.img_size
        self.n_pixels = self.original_frame.data.n_pixels
        self.display_order = self.original_frame.display_order
        self.coding_order = self.original_frame.coding_order
        self.seq_name = self.original_frame.seq_name

        # ----- Convert rate in bpp
        # Divide each entry of self.detailed_rate_nn by the number of pixel
        self.detailed_rate_nn_bpp: DescriptorCoolChic = {
            module_name: {
                weight_or_bias: rate_in_bits / self.n_pixels
                for weight_or_bias, rate_in_bits in module.items()
            }
            for module_name, module in self.detailed_rate_nn.items()
        }

        # ----- Copy all the quantities present in InterCodingModuleOutput
        quantities_from_inter_coding = [
            'alpha', 'beta', 'residue', 'flow_1', 'flow_2', 'prediction', 'masked_prediction'
        ]
        for k in quantities_from_inter_coding:
            if k in self.frame_encoder_output.additional_data:
                setattr(self, k, self.frame_encoder_output.additional_data.get(k))

        # ----- Compute several additional quantities
        if not self.alpha is None:
            self.alpha_mean = self.alpha.mean().item()

        if not self.beta is None:
            self.beta_mean = self.beta.mean().item()

        if not self.prediction is None:
            # Transform the reference to yuv 444 if needed
            if self.original_frame.data.frame_data_type == 'yuv420':
                original_frame_data = convert_420_to_444(self.original_frame.data.data)
            else:
                original_frame_data = self.original_frame.data.data

            self.prediction_psnr_db = -10 * torch.log10(compute_mse(self.prediction, original_frame_data))

            # Compute the dumbest prediction i.e. the average of the reference
            dummy_pred = torch.zeros_like(self.prediction)
            for ref in self.original_frame.refs_data:
                dummy_pred += ref.data
            dummy_pred /= len(self.original_frame.refs_data)

            self.dummy_prediction_psnr_db = -10 * torch.log10(compute_mse(dummy_pred, original_frame_data))

        # ------ Retrieve things related to the CoolChicEncoder from the additional
        # ------ outputs of the frame encoder.
        if 'detailed_rate_bit' in self.frame_encoder_output.additional_data:
            detailed_rate_bit = self.frame_encoder_output.additional_data.get('detailed_rate_bit')
            # Sum on the last three dimensions
            self.feature_rate_bpp = [
                x.sum(dim=(-1, -2, -3)) / (self.img_size[0] * self.img_size[1])
                for x in detailed_rate_bit
            ]

            upscaled_rate = []
            for rate in detailed_rate_bit:
                cur_c, cur_h, cur_w = rate.size()[-3:]

                # Ignore tensor with no channel
                if cur_c == 0:
                    continue

                # Rate is in bit, but since we're going to upsampling the rate values to match
                # the actual image size, we want to keep the total number of bit consistent.
                # To do so, we divide the rate by the upsampling ratio.
                # Example:
                # 2x2 feature maps with 8 bits for each sample gives a 4x4 visualisation
                # with 2 bits per sample. This make the total number of bits stay identical
                rate /=  (self.img_size[0] * self.img_size[1]) / (cur_h * cur_w)
                upscaled_rate.append(
                    nn.functional.interpolate(rate, size=self.img_size, mode='nearest')
                )

            upscaled_rate = torch.cat(upscaled_rate, dim=1)
            self.spatial_rate_bit = upscaled_rate.sum(dim=1, keepdim=True)

    def pretty_string(
        self,
        show_col_name: bool = False,
        mode: Literal['all', 'short'] = 'all',
        additional_data: Dict[str, Any] = {}
    ) -> str:
        """Return a pretty string formatting the data within the class.

        Args:
            show_col_name (bool, optional): True to also display col name. Defaults to False.
            mode (str, optional): Either "short" or "all". Defaults to 'all'.

        Returns:
            str: The formatted results
        """
        col_name = ''
        values = ''
        COL_WIDTH = 10
        INTER_COLUMN_SPACE = ' '

        for k in fields(self):
            if not self.should_be_printed(k.name, mode=mode):
                continue

            # ! Deep copying is needed but i don't know why?
            val = copy.deepcopy(getattr(self, k.name))

            if k.name == 'feature_rate_bpp':
                for i in range(len(val)):
                    col_name += f'{k.name + f"_{str(i).zfill(2)}":<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
                    values += f'{self.format_value(val[i], attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}'

            elif k.name == 'detailed_rate_nn_bpp':
                for subnetwork_name, subnetwork_detailed_rate in val.items():
                    col_name += f'{subnetwork_name + "_rate_bpp":<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
                    sum_weight_and_bias = sum([tmp for _, tmp in subnetwork_detailed_rate.items()])
                    values += f'{self.format_value(sum_weight_and_bias, attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}'

            else:
                col_name += f'{self.format_column_name(k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
                values += f'{self.format_value(val, attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}'

        for k, v in additional_data.items():
            col_name += f'{k:<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
            values += f'{v:<{COL_WIDTH}}{INTER_COLUMN_SPACE}'

        if show_col_name:
            return col_name + '\n' + values
        else:
            return values

    def should_be_printed(self, attribute_name: str, mode: str) -> bool:
        """Return True if the attribute named <attribute_name> should be printed
        in mode <mode>.

        Args:
            attribute_name (str): Candidate attribute to print
            mode (str): Either "short" or "all"

        Returns:
            bool: True if the attribute should be printed, False otherwise
        """

        # Syntax: {'attribute': [printed in mode xxx]}
        ATTRIBUTES = {
            # ----- This is printed in every modes
            'loss': ['short', 'all'],
            'psnr_db': ['short', 'all'],
            'total_rate_bpp': ['short', 'all'],
            'rate_latent_bpp': ['short', 'all'],
            'rate_nn_bpp': ['short', 'all'],
            'encoding_time_second': ['short', 'all'],
            'encoding_iterations_cnt': ['short', 'all'],

            # ----- This is only printed in mode all
            'alpha_mean': ['all'],
            'beta_mean': ['all'],
            'prediction_psnr_db': ['all'],
            'dummy_prediction_psnr_db': ['all'],
            'display_order': ['all'],
            'coding_order': ['all'],
            'lmbda': ['all'],
            'seq_name': ['all'],
            'feature_rate_bpp': ['all'],
            'detailed_rate_nn_bpp': ['all'],
            'ms_ssim_db': ['all'],
            'lpips_db': ['all'],
            'n_pixels': ['all'],
            'img_size': ['all'],
            'mac_decoded_pixel': ['all'],
        }

        if attribute_name not in ATTRIBUTES:
            return False

        if mode not in ATTRIBUTES.get(attribute_name):
            return False

        return True

    def format_value(
        self,
        value: Union[str, int, float, Tensor],
        attribute_name: str = ''
    ) -> str:

        if attribute_name == 'loss':
            value *= 1000

        if attribute_name == 'img_size':
            value = 'x'.join([str(tmp) for tmp in value])

        if isinstance(value, str):
            return value
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return f'{value:.6f}'
        elif isinstance(value, Tensor):
            return f'{value.item():.6f}'

    def format_column_name(self, col_name: str) -> str:

        # Syntax: {'long_name': 'short_name'}
        LONG_TO_SHORT = {
            'rate_latent_bpp': 'latent_bpp',
            'rate_nn_bpp': 'nn_bpp',
            'encoding_time_second': 'time_sec',
            'encoding_iterations_cnt': 'itr',

            'alpha_mean': 'alpha',
            'beta_mean': 'beta',
            'prediction_psnr_db': 'pred_db',
            'dummy_prediction_psnr_db': 'dummy_pred',

        }

        if col_name not in LONG_TO_SHORT:
            return col_name
        else:
            return LONG_TO_SHORT.get(col_name)


class FrameEncoder(nn.Module):
    def __init__(
        self,
        frame: Frame,
        coolchic_encoder_param: CoolChicEncoderParameter,
        frame_encoder_manager: FrameEncoderManager,
    ):
        """Create an encoder for the current frame.

        Args:
            frame (Frame): The frame to code.
            coolchic_encoder_param (CoolChicEncoderParameter): Parameters for the underlying
                CoolChicEncoder
            frame_encoder_manager (FrameEncoderManager): Encoding option for the current frame
        """
        super().__init__()

        # ----- Copy the parameters
        self.frame = frame
        self.coolchic_encoder_param = coolchic_encoder_param
        self.frame_encoder_manager = frame_encoder_manager

        # "Core" CoolChic codec. This will be reset by the warm-up function
        self.coolchic_encoder = CoolChicEncoder(self.coolchic_encoder_param)
        self.inter_coding_module = InterCodingModule(self.frame.frame_type)

    def forward(
        self,
        use_ste_quant: bool = False,
        AC_MAX_VAL: int = -1,
        flag_additional_outputs: bool = False,
    ) -> FrameEncoderOutput:
        """Perform the entire forward pass (i.e. one training step) of a video frame.

        Args:
            use_ste_quant (bool, optional): If True use the true quantization in the forward
                with its gradient set to self.param.ste_derivative (usually 0.01).
                Else, use the additive noise in the forward with its gradient set to 100.
                Defaults to False.
            AC_MAX_VAL (int, optional): If different from -1, clamp the value in
                [-AC_MAX_VAL ; AC_MAX_VAL + 1] to write the actual bitstream. Defaults to -1.
            flag_additional_outputs (bool, optional): True to fill FrameEncoderOutput.additional_data
                with many different quantities which can be used to analyze Cool-chic behavior.
                Defaults to False.

        Returns:
            FrameEncoderOutput: Output of the FrameEncoder for the forward pass. See above for
                additional details.
        """
        # CoolChic forward pass
        coolchic_encoder_output = self.coolchic_encoder(
            use_ste_quant=use_ste_quant, AC_MAX_VAL=AC_MAX_VAL, flag_additional_outputs=flag_additional_outputs
        )

        # Combine CoolChic output and reference frames through the inter coding modules
        inter_coding_output = self.inter_coding_module(
            coolchic_output=coolchic_encoder_output,
            references=[frame.data for frame in self.frame.refs_data],
            flag_additional_outputs=flag_additional_outputs
        )

        # Clamp decoded image & down sample YUV channel if needed
        if self.training:
            decoded_image = inter_coding_output.decoded_image
        else:
            max_dynamic = 2 ** (self.frame.data.bitdepth) - 1
            decoded_image = torch.round(inter_coding_output.decoded_image * max_dynamic) / max_dynamic

        if self.frame.data.frame_data_type == 'yuv420':
            decoded_image = convert_444_to_420(decoded_image)
            decoded_image = yuv_dict_clamp(decoded_image, min_val=0., max_val=1.)
        else:
            decoded_image = torch.clamp(decoded_image, 0., 1.)

        additional_data = {}
        if flag_additional_outputs:
            additional_data.update(coolchic_encoder_output.get('additional_data'))
            additional_data.update(inter_coding_output.additional_data)

        return FrameEncoderOutput(
            decoded_image=decoded_image,
            rate=coolchic_encoder_output.get('rate'),
            additional_data=additional_data
        )

    def set_to_train(self):
        self = self.train()
        self.coolchic_encoder = self.coolchic_encoder.train()
        self.inter_coding_module = self.inter_coding_module.train()

    def set_to_eval(self):
        self = self.eval()
        self.coolchic_encoder = self.coolchic_encoder.eval()
        self.inter_coding_module = self.inter_coding_module.eval()

    def loss_function(
        self,
        frame_encoder_out: FrameEncoderOutput,
        rate_mlp_bit: float = 0.,
        force_mse: bool = False,
        compute_logs: bool = False
    ) -> LossFunctionOutput:

        dist_weight = {'mse': 1.0, 'msssim': 0.0, 'lpips': 0.0} if force_mse else self.frame_encoder_manager.dist_weight

        msssim_norm_factor = 0.01
        lpips_norm_factor = 0.01

        # Compressed image is x_hat, original image is x.
        x_hat = frame_encoder_out.decoded_image
        x = self.frame.data.data

        if self.frame.data.frame_data_type == 'yuv420':
            device = x_hat.get('y').device
        else:
            device = x_hat.device

        # Compute MSE if needed i.e. if we need log or if required by the training metrics
        if compute_logs or dist_weight.get('mse') != 0:
            mse = compute_mse(x_hat, x)
        else:
            mse = torch.zeros((1), device=device)

        # Compute MSSSIM if needed i.e. if we need log or if required by the training metrics
        if compute_logs or dist_weight.get('msssim') != 0:
            msssim = compute_msssim(x_hat, x)
        else:
            msssim = torch.zeros((1), device=device)

        # # Compute LPIPS if needed i.e. if we need log or if required by the training metrics
        # if compute_logs or dist_weight.get('lpips') != 0:
        #     lpips = self.lpips(x_hat, x)
        # else:
        lpips = torch.ones((1), device=device)

        # Compute the distortion
        dist = dist_weight.get('mse') * mse + \
            dist_weight.get('msssim') * (1 - msssim) * msssim_norm_factor + \
            dist_weight.get('lpips') * lpips * lpips_norm_factor

        rate_bpp = (frame_encoder_out.rate.sum() + rate_mlp_bit) / self.frame.data.n_pixels

        # Final loss
        loss = dist + self.frame_encoder_manager.lmbda * rate_bpp

        # Construct the output module
        output = LossFunctionOutput(
            loss=loss,
            mse=mse.detach().item(),
            ms_ssim=msssim.detach().item(),
            lpips=lpips.detach().item(),
            rate_nn_bpp=rate_mlp_bit / self.frame.data.n_pixels,
            rate_latent_bpp=frame_encoder_out.rate.detach().sum().item() / self.frame.data.n_pixels,
        )

        return output

    @torch.no_grad()
    def test(self) -> FrameEncoderLogs:
        # 1. Get the rate associated to the network ----------------------------- #
        # The rate associated with the network is zero if it has not been quantize
        # before calling the test functions
        rate_mlp = 0.
        rate_per_module = self.coolchic_encoder.get_network_rate()
        for _, module_rate in rate_per_module.items():
            for _, param_rate in module_rate.items():   # weight, bias
                rate_mlp += param_rate

        # 2. Measure performance ------------------------------------------------ #
        self.set_to_eval()

        # flag_additional_outputs set to True to obtain more output
        frame_encoder_out = self.forward(use_ste_quant=False, AC_MAX_VAL=-1, flag_additional_outputs=True)

        loss_fn_output = self.loss_function(
            frame_encoder_out,
            rate_mlp_bit=rate_mlp,
            force_mse=False,
            compute_logs=True
        )

        encoder_logs = FrameEncoderLogs(
            loss_function_output=loss_fn_output,
            frame_encoder_output=frame_encoder_out,
            original_frame=self.frame,
            detailed_rate_nn=rate_per_module,
            quantization_param_nn=self.coolchic_encoder.get_network_quantization_step(),
            lmbda=self.frame_encoder_manager.lmbda,
            encoding_time_second=self.frame_encoder_manager.total_training_time_sec,
            encoding_iterations_cnt=self.frame_encoder_manager.iterations_counter,
            mac_decoded_pixel=self.coolchic_encoder.get_total_mac_per_pixel(),
        )

        # 3. Restore training mode ---------------------------------------------- #
        self.set_to_train()

        return encoder_logs

    def one_training_loop(
        self,
        device: POSSIBLE_DEVICE,
        frame_workdir: str,
        path_original_sequence: str,
        start_time: float,
        job_duration_min: float,
    ) -> TrainingExitCode:
        """Main training function of a FrameEncoder. It requires a frame_encoder_save_path
        in order to save the encoder periodically to allow for checkpoint.

        Args:
            device (POSSIBLE_DEVICE): On which device should the training run
            frame_encoder_save_path (str): Where to checkpoint the model
            path_original_sequence (str): Path to the raw .yuv file with the video to code.
                This should not really be seen by a FrameEncoder, but we need it to perform the
                inter warm-up where the references are shifted.
            start_time (float): Keep track of the when we started the overall training to
                requeue if need be
            job_duration_min (float): Exit and save the job after this duration is passed.
                Use -1 to only exit at the end of the entire encoding

        Returns:
            (TrainingExitCode): Exit code

        """
        # Loop until we've done all the required loops
        msg = '-' * 80 + '\n'
        msg += f'{" " * 30} Training loop {self.frame_encoder_manager.loop_counter + 1} / {self.frame_encoder_manager.n_loops}\n'
        msg += '-' * 80
        print(msg)

        self.to_device(device)

        if not self.frame_encoder_manager.warm_up_done:
            self.intra_warmup(device)
            self.to_device(device)

            # Inter warm up is only relevant for inter frame and must be called after the intra warm-up!
            if self.frame.frame_type in ['P', 'B']:
                self.inter_warmup(device, path_original_sequence)

            self.frame_encoder_manager.warm_up_done = True

        # Save model after checkpoint
        if is_job_over(start_time, max_duration_job_min=float(job_duration_min)):
            return TrainingExitCode.REQUEUE

        # Perform the successive training phase from phase_encoder_manager.phase_idx to
        # the total number of phase.
        # The counter phase_encoder_manager.phase_idx is incremented by the function
        # self.one_training_phase()
        for idx_phase in range(self.frame_encoder_manager.phase_idx, len(self.frame_encoder_manager.preset.all_phases)):
            print(f'{"-" * 30} Training phase: {idx_phase:>2} {"-" * 30}\n')
            self.one_training_phase(self.frame_encoder_manager.preset.all_phases[idx_phase])
            self.frame_encoder_manager.phase_idx += 1

            print(f'\nResults at the end of the phase:')
            print('--------------------------------')
            print(f'\n{self.test().pretty_string(show_col_name=True, mode="short")}\n')

            if is_job_over(start_time, max_duration_job_min=float(job_duration_min)):
                return TrainingExitCode.REQUEUE

        # At the end of each loop, compute the final loss
        frame_encoder_logs = self.test()

        # Write results file
        with open(f'{frame_workdir}results_loop_{self.frame_encoder_manager.loop_counter + 1}.tsv', 'w') as f_out:
            f_out.write(frame_encoder_logs.pretty_string(show_col_name=True, mode='all') + '\n')

        # We've beaten our record
        if self.frame_encoder_manager.record_beaten(frame_encoder_logs.loss):
            print(f'Best loss beaten at loop {self.frame_encoder_manager.loop_counter + 1}')
            print(f'Previous best loss: {self.frame_encoder_manager.best_loss * 1e3 :.6f}')
            print(f'New best loss     : {frame_encoder_logs.loss.cpu().item() * 1e3 :.6f}')

            self.frame_encoder_manager.set_best_loss(frame_encoder_logs.loss.cpu().item())

            # Save best results
            with open(f'{frame_workdir}results_best.tsv', 'w') as f_out:
                f_out.write(frame_encoder_logs.pretty_string(show_col_name=True, mode='all') + '\n')

            # # Generate the visualisation for the best frame encoder
            # self.generate_visualisation(f'{frame_workdir}')

        # Increment the loop counter, reset the warm up flag and the phase idx counter
        self.frame_encoder_manager.loop_counter += 1
        self.frame_encoder_manager.warm_up_done = False
        self.frame_encoder_manager.phase_idx = 0

        # We're done with this frame!
        return TrainingExitCode.END

    def intra_warmup(self, device: POSSIBLE_DEVICE):
        """/!\ Must be called **BEFORE** inter_warmup!

            Perform the "intra" warm-up i.e. N different mini training to select the best
        starting point. At the end of the warm-up, the starting point is registered
        as an attribute in self.coolchic_encoder.

        Args:
            device (POSSIBLE_DEVICE): The device on which the model should run.
        """
        start_time = time.time()

        training_preset = self.frame_encoder_manager.preset
        msg = '\nStarting intra warm up...'
        msg += f' Number of intra warm-up iterations: {training_preset.get_total_warmup_iterations()}\n'
        print(msg)

        _col_width = 14

        for idx_warmup_phase, warmup_phase in enumerate(training_preset.all_intra_warmups):
            print(f'{"-" * 30}  Warm-up phase: {idx_warmup_phase:>2} {"-" * 30}')

            # mem_info(f"Warmup-{idx_warmup_phase:02d}")

            # At the beginning of the first warmup phase, we must initialize all the models
            if idx_warmup_phase == 0:
                all_candidates = [
                    {
                        'model': CoolChicEncoder(self.coolchic_encoder_param),
                        'metrics': None,
                        'id': idx_model
                    }
                    for idx_model in range(warmup_phase.candidates)
                ]

            # At the beginning of the other warm-up phases, keep the desired number of best candidates
            else:
                all_candidates = all_candidates[:warmup_phase.candidates]

            # Construct the training phase object describing the options of this particular warm-up phase
            training_phase = TrainerPhase(
                lr=warmup_phase.lr,
                max_itr=warmup_phase.iterations,
                freq_valid=warmup_phase.freq_valid,
                start_temperature_softround=0.3,
                end_temperature_softround=0.3,
                start_kumaraswamy=2.0,
                end_kumaraswamy=2.0,
            )

            # ! idx_candidate is just the index of one candidate in the all_candidates list. It is **not** a
            # ! unique identifier for this candidate. This is given by:
            # !         all_candidates[idx_candidate].get('id')
            # ! the all_candidates list gives the ordered list of the best performing models so its order may change.
            for idx_candidate, candidate in enumerate(all_candidates):
                print(f'\nCandidate n° {idx_candidate:<2}, ID = {candidate.get("id"):<2}:')
                print(f'-------------------------\n')
                # mem_info(f"Warmup-cand-in {idx_warmup_phase:02d}-{idx_candidate:02d}")

                # Use the current candidate as our actual Cool-chic encoder
                self.coolchic_encoder = candidate.get('model')
                self.coolchic_encoder.to_device(device)

                # ! One training phase goes here!
                encoder_logs = self.one_training_phase(training_phase)

                self.coolchic_encoder.to_device('cpu')

                # Store the updated candidate on CPU
                all_candidates[idx_candidate] = {
                    'model': self.coolchic_encoder,
                    'metrics': encoder_logs,
                    'id': candidate.get('id')
                }
                # mem_info(f"Warmup-cand-out{idx_warmup_phase:02d}-{idx_candidate:02d}")

            # Sort all the models by ascending loss. The best one is all_candidates[0]
            all_candidates = sorted(all_candidates, key=lambda x: x.get('metrics').loss)

            # Print the results of this warm-up phase
            s = f'\n\nPerformance at the end of the warm-up phase:\n\n'
            s += f'{"ID":^{6}}|{"loss":^{_col_width}}|{"rate_bpp":^{_col_width}}|{"psnr_db":^{_col_width}}|\n'
            s += f'------|{"-" * _col_width}|{"-" * _col_width}|{"-" * _col_width}|\n'
            for candidate in all_candidates:
                s += f'{candidate.get("id"):^{6}}|'
                s += f'{candidate.get("metrics").loss.item() * 1e3:^{_col_width}.4f}|'
                s += f'{candidate.get("metrics").rate_latent_bpp:^{_col_width}.4f}|'
                s += f'{candidate.get("metrics").psnr_db:^{_col_width}.4f}|'
                s += '\n'
            print(s)

        # Keep only the best model
        best_model = all_candidates[0].get('model')
        self.coolchic_encoder = best_model

        # We've already worked for that many second during warm up
        warmup_duration =  time.time() - start_time

        print(f'Intra Warm-up is done!')
        print(f'Intra Warm-up time [s]: {warmup_duration:.2f}')
        print(f'Intra Winner ID       : {all_candidates[0].get("id")}\n')

    def inter_warmup(self, device: POSSIBLE_DEVICE, path_original_sequence: str):
        """ /!\ Must be called **AFTER** intra_warmup!

        Perform the "inter" warm-up, where the reference distance to the frame to code
        are progressively increased from 1 to the actual distance. This allows for a better
        convergence of the motion information. When we shift the reference, we use uncompressed
        frames since we do not have coded the required frame yet.

        Args:
            device (POSSIBLE_DEVICE): The device on which the model should run.
            path_original_sequence (str): Path to the raw .yuv file with the video to code.
                This should not really be seen by a FrameEncoder, but we need it to perform the
                inter warm-up where the references are shifted.
        """
        start_time = time.time()

        self.to_device(device)
        initial_encoder_logs = self.test()

        # Save the original module, we might need them later
        old_cool_chic_encoder_state_dict = OrderedDict(
            (k, v.detach().clone()) for k, v in self.coolchic_encoder.state_dict().items()
        )
        old_inter_coding_module = copy.deepcopy(self.inter_coding_module)

        inter_warmup_phase = self.frame_encoder_manager.preset.inter_warmup
        training_phase = TrainerPhase(
            lr=inter_warmup_phase.lr, max_itr=inter_warmup_phase.iterations
        )

        # Note: if we have two references there are at the same distance
        # to the frame to code.
        actual_ref_distance = self.frame.display_order - self.frame.index_references[0]
        actual_log2_ref_distance = int(math.log2(actual_ref_distance))

        msg = '\nStarting inter warm up...'
        msg += f' Number of inter warm-up iterations: {actual_log2_ref_distance * inter_warmup_phase.iterations}'
        print(msg)

        # Store the original frame (i.e. references and other stuff)
        actual_frame = copy.deepcopy(self.frame)

        # Successively moves the reference. We do at least one loop. Even if
        # the distance between the frame and its reference is 1.
        for log_ref_distance in range(actual_log2_ref_distance + 1):
            ref_distance = int(2 ** log_ref_distance)
            self.inter_coding_module.flow_gain = ref_distance

            msg = f'\nStep {log_ref_distance + 1:} / {actual_log2_ref_distance + 1} - Reference distance {ref_distance:>3}:\n'
            msg += '------------------------------------\n'
            print(msg)

            # ----- Set the references for the frame
            # The new index of the references
            new_ref_index = [self.frame.display_order - ref_distance]
            if self.frame.frame_type == 'B':
                new_ref_index.append(self.frame.display_order + ref_distance)
            self.frame.index_references = new_ref_index

            # Construct a list of two references located at -ref_distance and + ref_distance
            # of the current frame. Note that this is achieved by loading the original frame
            # data for the references instead of the compressed representation. We do this
            # because the compressed version of the references is not yet available.
            list_refs_data: List[FrameData] = [
                load_frame_data_from_file(path_original_sequence, idx_ref)
                for idx_ref in self.frame.index_references
            ]

            # Change references data format from 420 to 444
            if self.frame.data.frame_data_type == 'yuv420':
                list_refs_data = [
                    FrameData(
                        bitdepth=yuv420_frame_data.bitdepth,
                        frame_data_type='yuv444',
                        data=convert_420_to_444(yuv420_frame_data.data)
                    )
                    for yuv420_frame_data in list_refs_data
                ]

            self.frame.set_refs_data(list_refs_data)
            # To device is used to move the references to the required device
            self.to_device(device)

            # Perform one training phase to update the model and the results
            encoder_logs = self.one_training_phase(training_phase)

        if encoder_logs.loss <= initial_encoder_logs.loss:
            winner_str = 'Inter'
        else:
            winner_str = 'Intra'
            self.coolchic_encoder.load_state_dict(old_cool_chic_encoder_state_dict)
            self.inter_coding_module = old_inter_coding_module

        # Restore the real frame to code
        self.frame = actual_frame

        # We've already worked for that many second during warm up
        warmup_duration =  time.time() - start_time

        print(f'\nInter Warm-up is done!')
        print(f'Inter Warm-up time [s]: {warmup_duration:.2f}')
        print(f'\nWarm-up winner        : {winner_str}\n')

    def one_training_phase(self, trainer_phase: TrainerPhase):
        start_time = time.time()

        # ==== Keep track of the best loss and model for *THIS* current phase ==== #
        # Perform a first test to get the current best logs (it includes the loss)
        initial_encoder_logs = self.test()
        encoder_logs_best = initial_encoder_logs
        # ! Maybe self.cool_chic_encoder.state_dict()?
        this_phase_best_model = OrderedDict(
            (k, v.detach().clone()) for k, v in self.state_dict().items()
        )
        # ==== Keep track of the best loss and model for *THIS* current phase ==== #

        self.set_to_train()

        # =============== Build the list of parameters to optimize ============== #
        # Iteratively construct the list of required parameters... This is kind of a
        # strange syntax, which has been found quite empirically

        parameters_to_optimize = []

        if 'arm' in trainer_phase.optimized_module:
            parameters_to_optimize += [*self.coolchic_encoder.arm.parameters()]
        if 'upsampling' in trainer_phase.optimized_module:
            parameters_to_optimize += [*self.coolchic_encoder.upsampling.parameters()]
        if 'synthesis' in trainer_phase.optimized_module:
            parameters_to_optimize += [*self.coolchic_encoder.synthesis.parameters()]
        if 'latent' in trainer_phase.optimized_module:
            parameters_to_optimize += [*self.coolchic_encoder.latent_grids.parameters()]
        if 'all' in trainer_phase.optimized_module:
            parameters_to_optimize = self.parameters()

        optimizer = torch.optim.Adam(parameters_to_optimize, lr=trainer_phase.lr)
        # =============== Build the list of parameters to optimize ============== #

        scheduler = False
        # Scheduler for a single monotonic decrease of the learning rate from
        # trainer_phase.lr to 0.
        if trainer_phase.scheduling_period:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=trainer_phase.max_itr / trainer_phase.freq_valid,
                eta_min=0,
                last_epoch=-1,
                verbose=False
            )

        # Custom scheduling function for the soft rounding temperature and the kumaraswamy param
        def linear_schedule(initial_value, final_value, cur_itr, max_itr):
            return cur_itr * (final_value - initial_value) / max_itr + initial_value

        # Initialize soft rounding temperature and kumaraswamy parameter
        cur_tmp = linear_schedule(
            trainer_phase.start_temperature_softround,
            trainer_phase.end_temperature_softround,
            0,
            trainer_phase.max_itr
        )
        kumaraswamy_param = linear_schedule(
            trainer_phase.start_kumaraswamy,
            trainer_phase.end_kumaraswamy,
            0,
            trainer_phase.max_itr
        )

        self.coolchic_encoder.noise_quantizer.soft_round_temperature = cur_tmp
        self.coolchic_encoder.ste_quantizer.soft_round_temperature = cur_tmp
        self.coolchic_encoder.noise_quantizer.kumaraswamy_param = kumaraswamy_param

        cnt_record = 0
        show_col_name = True

        secure_with_best_model_period = trainer_phase.max_itr / 10

        # phase optimization
        for cnt in range(trainer_phase.max_itr):

            if cnt - cnt_record > trainer_phase.patience:

                # exceeding the patience level ends the phase
                break

            # This is slightly faster than optimizer.zero_grad()
            for param in self.parameters():
                param.grad = None

            # forward / backward
            out_forward = self.forward(use_ste_quant=trainer_phase.ste)
            loss_function_output = self.loss_function(out_forward)
            loss_function_output.loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1e-1, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()

            self.frame_encoder_manager.iterations_counter += 1

            # Each freq_valid iteration or at the end of the phase, compute validation loss and log stuff
            if ((cnt + 1) % trainer_phase.freq_valid == 0) or (cnt + 1 == trainer_phase.max_itr):
                #  a. Update iterations counter and training time and test model
                self.frame_encoder_manager.total_training_time_sec += time.time() - start_time
                start_time = time.time()

                # b. Test the model and check whether we've beaten our record
                encoder_logs = self.test()

                flag_new_record = False

                if encoder_logs.loss < encoder_logs_best.loss:
                    # A record must have at least -0.001 bpp or + 0.001 dB. A smaller improvement
                    # does not matter.
                    delta_psnr = encoder_logs.psnr_db - encoder_logs_best.psnr_db
                    delta_bpp  = encoder_logs.rate_latent_bpp - encoder_logs_best.rate_latent_bpp
                    flag_new_record = delta_bpp < 0.001 or delta_psnr > 0.001

                if flag_new_record:
                    # Save best model
                    for k, v in self.state_dict().items():
                        this_phase_best_model[k].copy_(v)

                    # ========================= reporting ========================= #
                    this_phase_loss_gain =  100 * (encoder_logs.loss - initial_encoder_logs.loss) / encoder_logs.loss
                    this_phase_psnr_gain =  encoder_logs.psnr_db - initial_encoder_logs.psnr_db
                    this_phase_bpp_gain =  encoder_logs.rate_latent_bpp - initial_encoder_logs.rate_latent_bpp

                    # log_new_record = f'{this_phase_loss_gain.item():+7.1f}% '
                    log_new_record = ''
                    log_new_record += f'{this_phase_bpp_gain:+6.3f} bpp '
                    log_new_record += f'{this_phase_psnr_gain:+6.3f} db'
                    # ========================= reporting ========================= #

                    # Update new record
                    encoder_logs_best = encoder_logs
                    cnt_record = cnt
                else:
                    log_new_record = ''

                # Show column name a single time
                additional_data = {
                    'STE': trainer_phase.ste,
                    'lr': f'{trainer_phase.lr if not scheduler else scheduler.get_last_lr()[0]:.8f}',
                    'optim': ','.join(trainer_phase.optimized_module),
                    'patience': (trainer_phase.patience - cnt + cnt_record) // trainer_phase.freq_valid,
                    'sr_temp': f'{cur_tmp:.5f}',
                    'kumara': f'{kumaraswamy_param:.5f}',
                    'record': log_new_record
                }
                print(
                    encoder_logs.pretty_string(
                        show_col_name=show_col_name,
                        mode='short',
                        additional_data=additional_data
                    )
                )
                show_col_name = False


                # Update soft rounding temperature and kumaraswamy noise
                cur_tmp = linear_schedule(
                    trainer_phase.start_temperature_softround,
                    trainer_phase.end_temperature_softround,
                    cnt,
                    trainer_phase.max_itr
                )
                kumaraswamy_param = linear_schedule(
                    trainer_phase.start_kumaraswamy,
                    trainer_phase.end_kumaraswamy,
                    cnt,
                    trainer_phase.max_itr
                )

                self.coolchic_encoder.noise_quantizer.soft_round_temperature = cur_tmp
                self.coolchic_encoder.ste_quantizer.soft_round_temperature = cur_tmp
                self.coolchic_encoder.noise_quantizer.kumaraswamy_param = kumaraswamy_param

                # Update scheduler
                if scheduler:
                    scheduler.step()

                # Restore training mode
                self.set_to_train()

            # the best model is periodically reloaded to avoid divergence
            if scheduler and (cnt % secure_with_best_model_period) == 0 and cnt:
                self.load_state_dict(this_phase_best_model)

        # Load best model found for this encoding loop
        self.load_state_dict(this_phase_best_model)

        # Quantize the model parameters at the end of the training phase
        if trainer_phase.quantize_model:
            self.quantize_model()

        # Final test to eventual retrieve the performance of the model
        encoder_logs = self.test()

        return encoder_logs

    @torch.no_grad()
    def quantize_model(self):
        """Quantize the current model, in place!.
        # ! We also obtain the integerized ARM here!"""

        start_time = time.time()
        self.set_to_eval()

        # We have to quantize all the modules that we want to send
        module_to_quantize = {
            module_name: getattr(self.coolchic_encoder, module_name)
            for module_name in self.coolchic_encoder.modules_to_send
        }

        best_q_step = {k: None for k in module_to_quantize}

        for module_name, module in module_to_quantize.items():
            # Start the RD optimization for the quantization step of each module with an
            # arbitrary high value for the RD cost.
            best_loss = 1e6

            # Save full precision parameters before quantizing
            module.save_full_precision_param()

            # Try to find the best quantization step
            all_q_step = module._POSSIBLE_Q_STEP
            for q_step_w, q_step_b in itertools.product(all_q_step, all_q_step):
                # Quantize
                current_q_step: DescriptorNN = {'weight': q_step_w, 'bias': q_step_b}
                quantization_success = module.quantize(current_q_step)

                if not quantization_success:
                    continue

                # Measure rate
                rate_per_module = module.measure_laplace_rate()
                total_rate_module_bit = sum([v for _, v in rate_per_module.items()])

                # Evaluate

                # ===================== Integerization of the ARM ===================== #
                if module_name == 'arm':
                    if ARMINT:
                        self.coolchic_encoder = self.coolchic_encoder.to_device('cpu')
                    module.set_quant(FIXED_POINT_FRACTIONAL_MULT)
                # ===================== Integerization of the ARM ===================== #

                frame_encoder_out = self.forward()

                # Compute results
                loss_function_output = self.loss_function(frame_encoder_out, total_rate_module_bit)

                # Store best quantization steps
                if loss_function_output.loss < best_loss:
                    best_loss = loss_function_output.loss
                    best_q_step[module_name] = current_q_step

            # Once we've tested all the possible quantization step: quantize one last
            # time with the best one we've found to actually use it.
            quantization_success = module.quantize(best_q_step[module_name])

            if not quantization_success:
                print(f'Greedy quantization failed!')
                sys.exit(0)

        print(f'\nTime greedy_quantization: {time.time() - start_time:4.1f} seconds\n')

        # Re-apply integerization of the module
        self.coolchic_encoder.arm.set_quant(FIXED_POINT_FRACTIONAL_MULT)

    def save(self) -> BytesIO:
        """Save the current CoolChicEncoder into a bytes buffer and return it.

        Returns:
            BytesIO: Bytes representing the saved coolchic model
        """
        buffer = BytesIO()
        data_to_save = {
            'frame_encoder_manager': self.frame_encoder_manager,
            'inter_coding_module': self.inter_coding_module,
            'coolchic_encoder': self.coolchic_encoder,
        }
        torch.save(data_to_save, buffer)
        return buffer

    def generate_visualisation(self, save_dir: str):
        """Generate different visualizations and put them inside save dir

        Args:
            save_dir (str): Absolute path of the directory where the visualizations
            should be stored
        """
        save_dir = save_dir.rstrip('/') + '/'
        subprocess.call(f'mkdir -p {save_dir}', shell=True)

        # Run the test to obtain the output
        frame_encoder_logs = self.test()

        visualisation_format = 'png' if self.frame.data.frame_data_type == 'rgb' else 'yuv'
        # =========================== Image & video coding visualisations =========================== #
        save_visualisation(
            FrameData(
                bitdepth=self.frame.data.bitdepth,
                frame_data_type=self.frame.data.frame_data_type,
                data=frame_encoder_logs.frame_encoder_output.decoded_image,
            ),
            f'{save_dir}decoded',
            mode='image',
            format=visualisation_format
        )

        save_visualisation(
            frame_encoder_logs.spatial_rate_bit, f'{save_dir}rate', mode='rate', format=visualisation_format
        )

        save_visualisation(
            frame_encoder_logs.frame_encoder_output.additional_data.get('detailed_sent_latent'),
            f'{save_dir}latent_sent',
            mode='feature',
            format=visualisation_format
        )
        save_visualisation(
            frame_encoder_logs.frame_encoder_output.additional_data.get('detailed_rate_bit'),
            f'{save_dir}rate_detailed',
            mode='rate',
            format=visualisation_format
        )
        save_visualisation(
            frame_encoder_logs.frame_encoder_output.additional_data.get('detailed_centered_latent'),
            f'{save_dir}latent_centered',
            mode='feature',
            format=visualisation_format
        )
        # =========================== Image & video coding visualisations =========================== #

        # ============================= Video coding only visualisations ============================ #
        if self.frame.data.frame_data_type != 'rgb':
            save_visualisation(
                FrameData(
                    bitdepth=self.frame.data.bitdepth,
                    frame_data_type='yuv444',
                    data=frame_encoder_logs.prediction,
                ),
                f'{save_dir}prediction',
                mode='image',
                format=visualisation_format
            )
            save_visualisation(
                FrameData(
                    bitdepth=self.frame.data.bitdepth,
                    frame_data_type='yuv444',
                    data=frame_encoder_logs.masked_prediction,
                ),
                f'{save_dir}masked_pred',
                mode='image',
                format=visualisation_format
            )
            save_visualisation(
                FrameData(
                    bitdepth=self.frame.data.bitdepth,
                    frame_data_type='yuv444',
                    # There is no proper way of visualizing the residue, Since
                    # it can be in [-1, 1], I just take the absolute value to obtain
                    # it in [0., 1.]
                    data=torch.clamp(frame_encoder_logs.residue.abs(), 0., 1.),
                ),
                f'{save_dir}residue',
                mode='image',
                format=visualisation_format
            )
            save_visualisation(frame_encoder_logs.flow_1, f'{save_dir}flow_1', mode='flow', format=visualisation_format)
            save_visualisation(frame_encoder_logs.flow_2, f'{save_dir}flow_2', mode='flow', format=visualisation_format)
            save_visualisation(frame_encoder_logs.alpha, f'{save_dir}alpha', mode='alpha', format=visualisation_format)
            save_visualisation(frame_encoder_logs.beta, f'{save_dir}beta', mode='beta', format=visualisation_format)
        # ============================= Video coding only visualisations ============================ #

    def to_device(self, device: POSSIBLE_DEVICE):
        """Push a model to a given device.

        Args:
            device (POSSIBLE_DEVICE): The device on which the model should run.
        """
        assert device in typing.get_args(POSSIBLE_DEVICE),\
            f'Unknown device {device}, should be in {typing.get_args(POSSIBLE_DEVICE)}'

        self = self.to(device)
        self.coolchic_encoder.to_device(device)
        if hasattr(self.frame, 'data'):
            self.frame.data.to_device(device)

        # Send all reference frames to the required device
        for i in range(len(self.frame.refs_data)):
            self.frame.refs_data[i].to_device(device)


def load_frame_encoder(raw_bytes: BytesIO, frame: Frame) -> FrameEncoder:
    """From already loaded raw bytes, load & return a CoolChicEncoder

    Args:
        raw_bytes (BytesIO): Already loaded raw bytes from which we'll instantiate
            the CoolChicEncoder.
        frame (Frame): Frame (and code references) corresponding to this
            FrameEncoder. This is not saved inside the frame_encoder.pt
            but rather provided by the parent VideoEncoder


    Returns:
        FrameEncoder: Frame encoder loaded by the function
    """
    # Reset the stream position to the beginning of the BytesIO object & load it
    raw_bytes.seek(0)
    loaded_data = torch.load(raw_bytes, map_location='cpu')

    coolchic_encoder = loaded_data['coolchic_encoder']

    # Create a frame encoder from the stored parameters
    frame_encoder = FrameEncoder(
        frame=frame,
        coolchic_encoder_param=coolchic_encoder.param,
        frame_encoder_manager=loaded_data['frame_encoder_manager'],

    )

    # Load the different submodules (i.e. CoolChic & the inter coding module)
    frame_encoder.coolchic_encoder = coolchic_encoder
    frame_encoder.inter_coding_module = loaded_data['inter_coding_module']

    return frame_encoder
