# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import copy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from enc.utils.manager import FrameEncoderManager
from enc.component.frame import FrameEncoder, FrameEncoderOutput
from enc.training.loss import (
    LossFunctionOutput,
    _compute_mse,
    loss_function,
)
from enc.utils.codingstructure import Frame, convert_420_to_444
from enc.utils.misc import DescriptorCoolChic
from torch import Tensor


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
    expgol_count_nn: DescriptorCoolChic             # Exp-Golomb count parameter for each NN weights & bias [ / ]

    lmbda: float                                    # Rate constraint in D + lambda * R [ / ]
    encoding_time_second: float                     # Duration of the encoding          [sec]
    encoding_iterations_cnt: int                    # Number of encoding iterations     [ / ]
    mac_decoded_pixel: float = 0.                   # Number of multiplication per decoded pixel

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
    detailed_rate_nn_bpp: DescriptorCoolChic = field(init=False)  # Rate for each NN weights & bias   [bpp]

    def __post_init__(self):
        # ----- Copy all the attributes of loss_function_output
        for f in fields(self.loss_function_output):
            setattr(self, f.name, getattr(self.loss_function_output, f.name))

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
            "alpha",
            "beta",
            "residue",
            "flow_1",
            "flow_2",
            "prediction",
            "masked_prediction",
        ]
        for k in quantities_from_inter_coding:
            if k in self.frame_encoder_output.additional_data:
                setattr(self, k, self.frame_encoder_output.additional_data.get(k))

        # ----- Compute several additional quantities
        if self.alpha is not None:
            self.alpha_mean = self.alpha.mean().item()

        if self.beta is not None:
            self.beta_mean = self.beta.mean().item()

        if self.prediction is not None:
            # Transform the reference to yuv 444 if needed
            if self.original_frame.data.frame_data_type == "yuv420":
                original_frame_data = convert_420_to_444(self.original_frame.data.data)
            else:
                original_frame_data = self.original_frame.data.data

            self.prediction_psnr_db = -10 * torch.log10(
                _compute_mse(self.prediction, original_frame_data)
            )

            # Compute the dumbest prediction i.e. the average of the reference
            dummy_pred = torch.zeros_like(self.prediction)
            for ref in self.original_frame.refs_data:
                dummy_pred += ref.data
            dummy_pred /= len(self.original_frame.refs_data)

            self.dummy_prediction_psnr_db = -10 * torch.log10(
                _compute_mse(dummy_pred, original_frame_data)
            )

        # ------ Retrieve things related to the CoolChicEncoder from the additional
        # ------ outputs of the frame encoder.
        if "detailed_rate_bit" in self.frame_encoder_output.additional_data:
            detailed_rate_bit = self.frame_encoder_output.additional_data.get(
                "detailed_rate_bit"
            )
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
                rate /= (self.img_size[0] * self.img_size[1]) / (cur_h * cur_w)
                upscaled_rate.append(
                    F.interpolate(rate, size=self.img_size, mode="nearest")
                )

            upscaled_rate = torch.cat(upscaled_rate, dim=1)
            self.spatial_rate_bit = upscaled_rate.sum(dim=1, keepdim=True)

    def pretty_string(
        self,
        show_col_name: bool = False,
        mode: Literal["all", "short"] = "all",
        additional_data: Dict[str, Any] = {},
    ) -> str:
        """Return a pretty string formatting the data within the class.

        Args:
            show_col_name (bool, optional): True to also display col name. Defaults to False.
            mode (str, optional): Either "short" or "all". Defaults to 'all'.

        Returns:
            str: The formatted results
        """
        col_name = ""
        values = ""
        COL_WIDTH = 10
        INTER_COLUMN_SPACE = " "

        for k in fields(self):
            if not self._should_be_printed(k.name, mode=mode):
                continue

            # ! Deep copying is needed but i don't know why?
            val = copy.deepcopy(getattr(self, k.name))
            if val is None:
                continue

            if k.name == "feature_rate_bpp":
                for i in range(len(val)):
                    col_name += f'{k.name + f"_{str(i).zfill(2)}":<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
                    values += f"{self._format_value(val[i], attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}"

            elif k.name == "detailed_rate_nn_bpp":
                for subnetwork_name, subnetwork_detailed_rate in val.items():
                    col_name += f'{subnetwork_name + "_rate_bpp":<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
                    sum_weight_and_bias = sum(
                        [tmp for _, tmp in subnetwork_detailed_rate.items()]
                    )
                    values += f"{self._format_value(sum_weight_and_bias, attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}"

            elif k.name == "quantization_param_nn":
                for subnetwork_name, subnetwork_detailed_q_step in val.items():
                    for tmp_k, tmp_val in subnetwork_detailed_q_step.items():
                        col_name += f'{subnetwork_name + "_" + tmp_k + "_q_step":<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
                        values += f"{self._format_value(tmp_val, attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}"

            elif k.name == "expgol_count_nn":
                for subnetwork_name, subnetwork_detailed_expgol_cnt in val.items():
                    for tmp_k, tmp_val in subnetwork_detailed_expgol_cnt.items():
                        col_name += f'{subnetwork_name + "_" + tmp_k + "_exp_cnt":<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
                        values += f"{self._format_value(tmp_val, attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}"

            else:
                col_name += f"{self._format_column_name(k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}"
                values += f"{self._format_value(val, attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}"

        for k, v in additional_data.items():
            col_name += f"{k:<{COL_WIDTH}}{INTER_COLUMN_SPACE}"
            values += f"{v:<{COL_WIDTH}}{INTER_COLUMN_SPACE}"

        if show_col_name:
            return col_name + "\n" + values
        else:
            return values

    def _should_be_printed(self, attribute_name: str, mode: str) -> bool:
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
            "loss": ["short", "all"],
            "psnr_db": ["short", "all"],
            "total_rate_bpp": ["short", "all"],
            "rate_latent_bpp": ["short", "all"],
            "rate_nn_bpp": ["short", "all"],
            "encoding_time_second": ["short", "all"],
            "encoding_iterations_cnt": ["short", "all"],
            # ----- This is only printed in mode all
            "alpha_mean": ["all"],
            "beta_mean": ["all"],
            "prediction_psnr_db": ["all"],
            "dummy_prediction_psnr_db": ["all"],
            "display_order": ["all"],
            "coding_order": ["all"],
            "lmbda": ["all"],
            "seq_name": ["all"],
            "feature_rate_bpp": ["all"],
            "detailed_rate_nn_bpp": ["all"],
            "ms_ssim_db": ["all"],
            "lpips_db": ["all"],
            "n_pixels": ["all"],
            "img_size": ["all"],
            "mac_decoded_pixel": ["all"],
            "quantization_param_nn": ["all"],
            "expgol_count_nn": ["all"],
        }

        if attribute_name not in ATTRIBUTES:
            return False

        if mode not in ATTRIBUTES.get(attribute_name):
            return False

        return True

    def _format_value(
        self, value: Union[str, int, float, Tensor], attribute_name: str = ""
    ) -> str:
        if attribute_name == "loss":
            value *= 1000

        if attribute_name == "img_size":
            value = "x".join([str(tmp) for tmp in value])

        if isinstance(value, str):
            return value
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return f"{value:.6f}"
        elif isinstance(value, Tensor):
            return f"{value.item():.6f}"

    def _format_column_name(self, col_name: str) -> str:
        # Syntax: {'long_name': 'short_name'}
        LONG_TO_SHORT = {
            "rate_latent_bpp": "latent_bpp",
            "rate_nn_bpp": "nn_bpp",
            "encoding_time_second": "time_sec",
            "encoding_iterations_cnt": "itr",
            "alpha_mean": "alpha",
            "beta_mean": "beta",
            "prediction_psnr_db": "pred_db",
            "dummy_prediction_psnr_db": "dummy_pred",
        }

        if col_name not in LONG_TO_SHORT:
            return col_name
        else:
            return LONG_TO_SHORT.get(col_name)


@torch.no_grad()
def test(
    frame_encoder: FrameEncoder,
    frame: Frame,
    frame_encoder_manager: FrameEncoderManager,
) -> FrameEncoderLogs:
    """Evaluate the performance of a ``FrameEncoder`` when encoding a ``Frame``.

    Args:
        frame_encoder: FrameEncoder to be evaluated.
        frame: The original frame to compress. It provides both the
            target (original non compressed frame) as well as the reference(s)
            (list of already decoded images)
        lambda: Rate constraint lambda. Only requires to compute a meaningfull
            loss :math:`\\mathcal{L} = \\mathrm{D} + \\lambda \\mathrm{R}`
        frame_encoder_manager: Contains (among other things) the rate constraint
            :math:`\\lambda`. It is also used to track the total encoding time
            and encoding iterations.

    Returns:
        Many logs on the performance of the FrameEncoder.
        See doc of ``FrameEncoderLogs``.
    """
    # 1. Get the rate associated to the network ----------------------------- #
    # The rate associated with the network is zero if it has not been quantize
    # before calling the test functions
    rate_mlp = 0.0
    rate_per_module = frame_encoder.coolchic_encoder.get_network_rate()
    for _, module_rate in rate_per_module.items():
        for _, param_rate in module_rate.items():  # weight, bias
            rate_mlp += param_rate

    # 2. Measure performance ------------------------------------------------ #
    frame_encoder.set_to_eval()

    # flag_additional_outputs set to True to obtain more output
    frame_encoder_out = frame_encoder.forward(
        reference_frames=[ref_i.data for ref_i in frame.refs_data],
        quantizer_noise_type="none",
        quantizer_type="hardround",
        AC_MAX_VAL=-1,
        flag_additional_outputs=True,
    )

    loss_fn_output = loss_function(
        frame_encoder_out.decoded_image,
        frame_encoder_out.rate,
        frame.data.data,
        lmbda=frame_encoder_manager.lmbda,
        rate_mlp_bit=rate_mlp,
        compute_logs=True,
    )

    encoder_logs = FrameEncoderLogs(
        loss_function_output=loss_fn_output,
        frame_encoder_output=frame_encoder_out,
        original_frame=frame,
        detailed_rate_nn=rate_per_module,
        quantization_param_nn=frame_encoder.coolchic_encoder.get_network_quantization_step(),
        expgol_count_nn=frame_encoder.coolchic_encoder.get_network_expgol_count(),
        encoding_time_second=frame_encoder_manager.total_training_time_sec,
        encoding_iterations_cnt=frame_encoder_manager.iterations_counter,
        mac_decoded_pixel=frame_encoder.coolchic_encoder.get_total_mac_per_pixel(),
        lmbda=frame_encoder_manager.lmbda
    )

    # 3. Restore training mode ---------------------------------------------- #
    frame_encoder.set_to_train()

    return encoder_logs
