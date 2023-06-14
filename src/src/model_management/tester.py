# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass, field
from model_management.loss import loss_fn
from models.cool_chic import CoolChicEncoder

from utils.constants import ARMINT


@dataclass
class CoolChicEncoderLogs():
    """Everything that should be returned and measured by the test function."""
    rate_latent_bpp: float          # Rate of the y latent in bit per pixel     [bpp]
    rate_arm_bpp: float             # Rate of the ARM MLP                       [bpp]
    rate_synthesis_bpp: float       # Rate of the synthesis MLP                 [bpp]
    rate_upsampling_bpp: float      # Rate of the Latent upsampling module      [bpp]

    mse: float                      # MSE of the decoded image                  [ / ]
    psnr_db: float                  # PSNR of the decoded image                 [ dB]
    ms_ssim_db: float               # -10log(1 - MS-SSIM) decoded img           [ dB]
    loss: float                     # D + lambda * R                            [ / ]
    q_step_arm_weight: float        # Q step for the ARM MLP weights            [ / ]
    q_step_arm_bias: float          # Q step for the ARM MLP biases             [ / ]
    q_step_synthesis_weight: float  # Q step for the synthesis MLP weights      [ / ]
    q_step_synthesis_bias: float    # Q step for the synthesis MLP biases       [ / ]
    q_step_upsampling_weight: float # Q step for the upsampling MLP weights     [ / ]
    q_step_upsampling_bias: float   # Q step for the upsampling MLP biases      [ / ]
    mac_decoded_pixel: float        # Number of MAC / decoded pixel             [ / ]
    img_size: Tuple[int, int]       # Image size (height, width)                [ / ]
    lmbda: float                    # Loss = D + lambda R                       [ / ]
    iterations: int                 # Total number of iterations done           [ / ]
    time_sec: float                 # Total training time                       [sec]
    rate_per_latent_bpp: List[float]    # Rate of each individual feature map   [bpp]
    seq_name: str = ''              # Sequence name                             [ / ]

    # ==================== Not set by the init function ===================== #
    rate_mlp_bpp: float = field(init=False)         # Rate of all MLPs          [bpp]
    rate_bpp: float = field(init=False)             # Total rate                [bpp]
    share_rate_latent: float = field(init=False)    # Latent share in total rate[ % ]
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        self.rate_mlp_bpp = self.rate_arm_bpp + self.rate_synthesis_bpp + self.rate_upsampling_bpp


        self.rate_bpp = self.rate_mlp_bpp + self.rate_latent_bpp

        if self.rate_bpp==0:
            self.share_rate_latent = 0
        else:
            self.share_rate_latent = self.rate_latent_bpp / self.rate_bpp

    def to_string(
        self,
        mode: str = 'short',
        additional_data: Optional[Dict[str, float]] = None,
        print_col_name: bool = False,
    ) -> str:
        """Return a pretty string containing the information stored into this object.

        Args:
            mode (str, optional): Either "short", "more" or "all". Defaults to "short".
            additional_data (Dict[str, float], optional):
                Additional data to print, not stored into a CoolChicEncoderLogs.
                Defaults to None
            print_col_name (bool, optional): If False only print the values.
                Defaults to False

        Returns:
            str: All the information about a CoolChicEncoder performance (and more!)
        """
        # Construct the list of values to print
        keys = ['loss', 'psnr_db', 'rate_bpp', 'iterations', 'time_sec', 'rate_mlp_bpp', 'ms_ssim_db']
        if mode in ['more', 'all']:
            # keys += ['ms_ssim_db']
            pass
        if mode == 'all':
            keys += ['img_size', 'lmbda', 'mac_decoded_pixel', 'share_rate_latent', 'rate_arm_bpp', 'rate_synthesis_bpp', 'rate_upsampling_bpp', 'rate_per_latent_bpp']

        ALL_FIELDS_PRECISION = {
            'loss': '.6f',
            'psnr_db': '.6f',
            'rate_bpp': '.6f',
            'iterations': '.0f',
            'time_sec': '.2f',
            'rate_mlp_bpp': '.6f',
            'ms_ssim_db': '.6f',
        }

        if mode in ['more', 'all']:
            # ALL_FIELDS_PRECISION['ms_ssim_db'] = '.6f'
            pass

        if mode in ['all']:
            ALL_FIELDS_PRECISION['img_size'] = 'no_formatting'
            ALL_FIELDS_PRECISION['lmbda'] = '.6f'
            ALL_FIELDS_PRECISION['mac_decoded_pixel'] = '.0f'
            ALL_FIELDS_PRECISION['share_rate_latent'] = '.3f'
            ALL_FIELDS_PRECISION['rate_arm_bpp'] = '.6f'
            ALL_FIELDS_PRECISION['rate_synthesis_bpp'] = '.6f'
            ALL_FIELDS_PRECISION['rate_upsampling_bpp'] = '.6f'
            ALL_FIELDS_PRECISION['rate_per_latent_bpp'] = '.6f'

        # Print them
        str_keys, str_vals = '', ''
        MIN_COL_WIDTH = 14
        for field_name, field_formatting in ALL_FIELDS_PRECISION.items():
            # Special case as this is a list
            if field_name == 'rate_per_latent_bpp':
                for idx_lat in range(len(self.__getattribute__('rate_per_latent_bpp'))):
                    cur_key = f'rate_latent_{idx_lat}_bpp'
                    col_width = max(len(cur_key) + 3, MIN_COL_WIDTH)
                    str_keys += f'{cur_key:<{col_width}}'

                    v = self.__getattribute__('rate_per_latent_bpp')[idx_lat].item()
                    val_to_print = f'{float(v):{field_formatting}}'
                    str_vals += f'{val_to_print:<{col_width}}'
                continue

            col_width = max(len(field_name) + 3, MIN_COL_WIDTH)
            str_keys += f'{field_name:<{col_width}}'
            v = self.__getattribute__(field_name)
            v = v.item() if isinstance(v, Tensor) else v


            if isinstance(v, torch.Size):
                val_to_print = ','.join([str(x) for x in v])
            else:
                # Check if v is a number:
                # The string formatting here allows to deal with small value being change
                # to scientific notation
                #
                #                   |
                #                   |
                #                   v
                is_number = str(f'{v:.20f}').replace(".", "", 1).isdigit()

                # ! Rename into val_to_print to avoid inplace modification of the value
                if is_number:
                    # ! milli-loss otherwise we don't see anything
                    val_to_print = v * 1000 if 'loss' in field_name else v
                    if isinstance(val_to_print, int):
                        val_to_print = f'{val_to_print}'
                    else:
                        val_to_print = f'{float(val_to_print):{field_formatting}}'

                else:
                    val_to_print = v

            str_vals += f'{val_to_print:<{col_width}}'

        # If we have more things to print
        if additional_data is not None:
            for k, v in additional_data.items():
                col_width = max(len(k) + 3, MIN_COL_WIDTH)
                str_keys += f'{k:<{col_width}}'

                if isinstance(v, List):
                    v = '-'.join(v)

                # Check if v is a number:
                is_number = str(v).replace(".", "", 1).isdigit()
                if is_number:
                    if isinstance(v, int):
                        v = f'{v}'
                    else:
                        v = f'{float(v):.4f}'
                str_vals += f'{v:<{col_width}}'

        res = str_keys + '\n' + str_vals if print_col_name else str_vals
        return res


@torch.no_grad()
def test(model: CoolChicEncoder) -> CoolChicEncoderLogs:
    """Perform the evaluation of a CoolChicEncoder.

    Args:
        model (CoolChicEncoder): A CoolChicEncoder to test

    Returns:
        CoolChicEncoderLogs: Encoder-side results
    """
    # 1. Get the rate associated to the network ----------------------------- #
    # The rate associated with the network is zero if it has not been quantize
    # before calling the test functions
    rate_model = model.get_network_rate()
    total_rate_model_bits = 0.

    for _, module_rate in rate_model.items():       # arm, synthesis, upsampling
        for _, param_rate in module_rate.items():   # weight, bias
            total_rate_model_bits += param_rate

    # 2. Measure performance ------------------------------------------------ #
    model = model.eval()
    # Visu set to True to obtain detailed rate per latent
    model_out = model(visu=True)

    target_img = model.param.img.to('cpu') if ARMINT else model.param.img
    _, metrics = loss_fn(
        model_out,
        target_img,
        model.param.lmbda,
        compute_logs=True,
        dist_mode=model.param.dist,
        rate_mlp=total_rate_model_bits
    )

    # 3. Construct the output CoolChicEncoderLogs --------------------------- #
    n_pixels = model.param.img_size[0] * model.param.img_size[1]

    arm_q_step = model.arm.get_q_step()
    synthesis_q_step = model.synthesis.get_q_step()
    upsampling_q_step = model.upsampling.get_q_step()

    res = CoolChicEncoderLogs(  # type: ignore
        rate_latent_bpp=metrics.get('rate_y'),
        rate_arm_bpp=sum([v for _, v in rate_model.get('arm').items()]) / n_pixels,
        rate_synthesis_bpp=sum([v for _, v in rate_model.get('synthesis').items()]) / n_pixels,
        rate_upsampling_bpp=sum([v for _, v in rate_model.get('upsampling').items()]) / n_pixels,
        mse=metrics.get('mse'),
        psnr_db=metrics.get('psnr'),
        ms_ssim_db=metrics.get('ms_ssim_db'),
        loss=metrics.get('loss'),
        q_step_arm_weight=0. if arm_q_step is None else arm_q_step['weight'],
        q_step_arm_bias=0. if arm_q_step is None else arm_q_step['bias'],
        q_step_synthesis_weight=0. if synthesis_q_step is None else synthesis_q_step['weight'],
        q_step_synthesis_bias=0. if synthesis_q_step is None else synthesis_q_step['bias'],
        q_step_upsampling_weight=0. if upsampling_q_step is None else upsampling_q_step['weight'],
        q_step_upsampling_bias=0. if upsampling_q_step is None else upsampling_q_step['bias'],
        mac_decoded_pixel=model.get_total_mac_per_pixel(),
        img_size=model.param.img_size,
        lmbda=model.param.lmbda,
        time_sec=model.total_training_time_sec,
        iterations=model.iterations_counter,
        rate_per_latent_bpp=metrics.get('rate_per_latent_bpp')
    )

    return res
