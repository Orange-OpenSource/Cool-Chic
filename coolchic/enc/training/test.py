# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import copy
import typing
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from enc.component.types import DescriptorCoolChic, DescriptorNN
import torch
import torch.nn.functional as F
from enc.component.frame import FrameEncoder, FrameEncoderOutput, NAME_COOLCHIC_ENC
from enc.io.format.yuv import convert_420_to_444
from enc.training.loss import (
    LossFunctionOutput,
    _compute_mse,
    loss_function,
    )
from enc.utils.codingstructure import Frame
from enc.training.manager import FrameEncoderManager
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

    loss_function_output: LossFunctionOutput    # All outputs from the loss function, will be copied is __post_init__
    frame_encoder_output: FrameEncoderOutput    # Output of frame encoder forward
    original_frame: Frame                       # Non coded frame

    detailed_rate_nn: Dict[str, DescriptorCoolChic]         # Rate for each NN weights & bias   [bit]
    quantization_param_nn: Dict[str, DescriptorCoolChic]    # Quantization step for each NN weights & bias [ / ]
    expgol_count_nn: Dict[str, DescriptorCoolChic]          # Exp-Golomb count parameter for each NN weights & bias [ / ]

    lmbda: float  # Rate constraint in D + lambda * R [ / ]
    encoding_time_second: float  # Duration of the encoding          [sec]
    encoding_iterations_cnt: int  # Number of encoding iterations     [ / ]
    mac_decoded_pixel: float = 0.0  # Number of multiplication per decoded pixel

    # ==================== Not set by the init function ===================== #
    # Everything here is derived from frame_encoder_output and original_frame

    # ----- CoolChicEncoder outputs
    # Spatial distribution of the rate, obtained by summing the rate of the different features
    # for each spatial location (in bit). [1, 1, H, W]
    spatial_rate_bit: Optional[Tensor] = field(init=False)
    # Feature distribution of the rate, obtained by the summing all the spatial location
    # of a given feature. [Number of latent resolution]
    feature_rate_bpp: Optional[List[float]] = field(
        init=False, default_factory=lambda: []
    )

    # ----- Computed from loss_function_output.rate_latent_bpp
    rate_latent_residue_bpp: float = field(init=False)
    rate_latent_motion_bpp: float = field(init=False)

    # ----- Inter coding module outputs
    alpha: Optional[Tensor] = field(init=False, default=None)   # Inter / intra switch
    beta: Optional[Tensor] = field(init=False, default=None)    # Bi-directional prediction weighting
    residue: Optional[Tensor] = field(init=False, default=None) # Residue
    flow_1: Optional[Tensor] = field(init=False, default=None)  # Optical flow for the first reference
    flow_2: Optional[Tensor] = field(init=False, default=None)  # Optical flow for the second reference
    pred: Optional[Tensor] = field(init=False, default=None)  # Temporal prediction
    masked_pred: Optional[Tensor] = field(init=False, default=None)  # Temporal prediction * alpha

    # ----- Compute prediction performance
    alpha_mean: Optional[float] = field(init=False, default=None)  # Mean value of alpha
    beta_mean: Optional[float] = field(init=False, default=None)  # Mean value of beta
    pred_psnr_db: Optional[float] = field(init=False, default=None)  # PSNR of the prediction
    dummy_pred_psnr_db: Optional[float] = field(init=False, default=None)  # PSNR of a prediction if we had no motion

    # ----- Miscellaneous quantities recovered from self.frame
    img_size: Tuple[int, int] = field(init=False)  # [Height, Width]
    n_pixels: int = field(init=False)  # Height x Width
    display_order: int = field(init=False)  # Index of the current frame in display order
    coding_order: int = field(init=False)  # Index of the current frame in coding order
    frame_offset: int = field(init=False, default=0)  # Skip the first <frame_offset> frames of the video
    seq_name: str = field(init=False)  # Name of the sequence to which this frame belong

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
        self.frame_offset = self.original_frame.frame_offset
        self.seq_name = self.original_frame.seq_name

        # ----- Information related to nn net rate / quantization & exp-golomb
        #
        # Iterates on all the possible names for a cool-chic encoder. If we find
        # something, we keep the value, otherwise we put a O so that all fields
        # are always filled.

        # Divide each entry of self.detailed_rate_nn by the number of pixel
        self.detailed_rate_nn_bpp: Dict[NAME_COOLCHIC_ENC, DescriptorCoolChic] = {}

        # Loop on all possible cool-chic encoder name
        for cc_name in typing.get_args(NAME_COOLCHIC_ENC):
            # Loop on all neural networks composing a Cool-chic Encoder
            self.detailed_rate_nn_bpp[cc_name] = {}

            for nn_name in [x.name for x in fields(DescriptorCoolChic)]:
                self.detailed_rate_nn_bpp[cc_name][nn_name] = {}

                # Loop on weight and biases
                for weight_or_bias in [x.name for x in fields(DescriptorNN)]:
                    # Convert the value to bpp if we have one
                    if cc_name in self.detailed_rate_nn and nn_name in self.detailed_rate_nn[cc_name]:
                        self.detailed_rate_nn_bpp[cc_name][nn_name][weight_or_bias] = (
                            self.detailed_rate_nn[cc_name][nn_name][weight_or_bias]
                            / self.n_pixels
                        )
                    # Else we set it to 0
                    else:
                        self.detailed_rate_nn_bpp[cc_name][nn_name][weight_or_bias] = 0

        # ----- Get sum of latent rate for each cool-chic encoder
        for cc_name in typing.get_args(NAME_COOLCHIC_ENC):
            setattr(
                self, f"rate_latent_{cc_name}_bpp", self.rate_latent_bpp.get(cc_name, 0)
            )

        # ----- Copy all the quantities present in InterCodingModuleOutput
        quantities_from_inter_coding = [
            "alpha",
            "beta",
            "residue",
            "flow_1",
            "flow_2",
            "pred",
            "masked_pred",
        ]
        for k in quantities_from_inter_coding:
            if k in self.frame_encoder_output.additional_data:
                setattr(self, k, self.frame_encoder_output.additional_data.get(k))

        # ----- Compute several additional quantities
        if self.alpha is not None:
            self.alpha_mean = self.alpha.mean().item()
        # No alpha for intra frame
        else:
            self.alpha_mean = 0

        if self.beta is not None:
            self.beta_mean = self.beta.mean().item()
        # No beta for I & P frames
        else:
            self.beta_mean = 0

        if self.pred is not None:
            # Transform the reference to yuv 444 if needed
            if self.original_frame.data.frame_data_type == "yuv420":
                original_frame_data = convert_420_to_444(self.original_frame.data.data)
            else:
                original_frame_data = self.original_frame.data.data

            self.pred_psnr_db = -10 * torch.log10(
                _compute_mse(self.pred, original_frame_data) + 1e-10
            )

            # Compute the dumbest prediction i.e. the average of the reference
            dummy_pred = torch.zeros_like(self.pred)
            for ref in self.original_frame.refs_data:
                dummy_pred += ref.data
            dummy_pred /= len(self.original_frame.refs_data)

            self.dummy_pred_psnr_db = -10 * torch.log10(
                _compute_mse(dummy_pred, original_frame_data) + 1e-10
            )
        # No prediction for intra frame.
        else:
            self.pred_psnr_db = 0
            self.dummy_pred_psnr_db = 0

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

            elif k.name == "detailed_dist_db":
                for dist_name, dist_val in self.detailed_dist_db.items():
                    col_name += f'{dist_name:<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
                    values += f"{self._format_value(dist_val, attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}"

            # This concerns the different neural networks composing the
            # different cool-chic encoders.
            elif k.name in [
                "detailed_rate_nn_bpp",
                "quantization_param_nn",
                "expgol_count_nn",
            ]:
                match k.name:
                    case "detailed_rate_nn_bpp":
                        col_name_sufix = "_rate_bpp"
                    case "quantization_param_nn":
                        col_name_sufix = "_q_step"
                    case "expgol_count_nn":
                        col_name_sufix = "_exp_cnt"
                    case _:
                        pass

                # Loop on all possible cool-chic encoder name
                for cc_name in typing.get_args(NAME_COOLCHIC_ENC):
                    # Loop on all neural networks composing a Cool-chic Encoder
                    for nn_name in [x.name for x in fields(DescriptorCoolChic)]:
                        for weight_or_bias in [x.name for x in fields(DescriptorNN)]:
                            tmp_col_name = (
                                cc_name
                                + "_"
                                + nn_name
                                + "_"
                                + weight_or_bias
                                + col_name_sufix
                            )
                            col_name += (
                                f"{tmp_col_name:<{COL_WIDTH}}{INTER_COLUMN_SPACE}"
                            )

                            try:
                                tmp_val = val[cc_name][nn_name][weight_or_bias]
                            except KeyError:
                                tmp_val = 0

                            tmp_val = self._format_value(tmp_val, attribute_name=k.name)
                            values += f"{tmp_val:<{COL_WIDTH}}{INTER_COLUMN_SPACE}"

            # Default case
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
            "dist_db": ["short", "all"],
            "detailed_dist_db": ["short", "all"],
            "total_rate_bpp": ["short", "all"],
            "total_rate_latent_bpp": ["short", "all"],
            "total_rate_nn_bpp": ["short", "all"],
            "encoding_time_second": ["short", "all"],
            "encoding_iterations_cnt": ["short", "all"],
            # ----- This is only printed in mode all
            "alpha_mean": ["all"],
            "beta_mean": ["all"],
            "pred_psnr_db": ["all"],
            "dummy_pred_psnr_db": ["all"],
            "display_order": ["all"],
            "coding_order": ["all"],
            "frame_offset": ["all"],
            "lmbda": ["all"],
            "seq_name": ["all"],
            "feature_rate_bpp": ["all"],
            "detailed_rate_nn_bpp": ["all"],
            "n_pixels": ["all"],
            "img_size": ["all"],
            "mac_decoded_pixel": ["all"],
            # Uncomment to log the quantization step and exp-golomb param
            # "quantization_param_nn": ["all"],
            # "expgol_count_nn": ["all"],
        }

        for cc_name in typing.get_args(NAME_COOLCHIC_ENC):
            ATTRIBUTES[f"rate_latent_{cc_name}_bpp"] = ["all"]

        # Add a few quantities for some particular frame type
        if self.original_frame.frame_type != "I":
            ATTRIBUTES["rate_latent_residue_bpp"] = ["all", "short"]
            ATTRIBUTES["rate_latent_motion_bpp"] = ["all", "short"]

            ATTRIBUTES["alpha_mean"].append("short")
            ATTRIBUTES["pred_psnr_db"].append("short")
            ATTRIBUTES["dummy_pred_psnr_db"].append("short")
        if self.original_frame.frame_type == "B":
            ATTRIBUTES["beta_mean"].append("short")

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

        # Number of digits after the decimal point
        DEFAULT_ACCURACY = "6"
        ACCURACY = {
            "alpha_mean": "3",
            "beta_mean": "3",
            "pred_psnr_db": "3",
            "dummy_pred_psnr_db": "3",
            "encoding_time_second": "1",
        }
        cur_accuracy = ACCURACY.get(attribute_name, DEFAULT_ACCURACY)

        if isinstance(value, str):
            return value
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return f"{value:.{cur_accuracy}f}"
        elif isinstance(value, Tensor):
            return f"{value.item():.{cur_accuracy}f}"

    def _format_column_name(self, col_name: str) -> str:
        # Syntax: {'long_name': 'short_name'}
        LONG_TO_SHORT = {
            "total_rate_latent_bpp": "latent_bpp",
            "total_rate_nn_bpp": "nn_bpp",
            "total_rate_bpp": "rate_bpp",
            "encoding_time_second": "time_sec",
            "encoding_iterations_cnt": "itr",
            "alpha_mean": "alpha",
            "beta_mean": "beta",
            "pred_psnr_db": "pred_db",
            "dummy_pred_psnr_db": "dummy_pred",
            "rate_latent_residue_bpp": "residue_bpp",
            "rate_latent_motion_bpp": "motion_bpp",
            "dist_db": "dist_db",
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
    detailed_rate_nn, total_rate_nn_bit = frame_encoder.get_network_rate()

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
        decoded_image=frame_encoder_out.decoded_image,
        rate_latent_bit=frame_encoder_out.rate,
        target_image=frame.data.data,
        dist_weight=frame_encoder_manager.dist_weight,
        lmbda=frame_encoder_manager.lmbda,
        total_rate_nn_bit=total_rate_nn_bit,
        compute_logs=True,
    )

    encoder_logs = FrameEncoderLogs(
        loss_function_output=loss_fn_output,
        frame_encoder_output=frame_encoder_out,
        original_frame=frame,
        detailed_rate_nn=detailed_rate_nn,
        quantization_param_nn=frame_encoder.get_network_quantization_step(),
        expgol_count_nn=frame_encoder.get_network_expgol_count(),
        encoding_time_second=frame_encoder_manager.total_training_time_sec,
        encoding_iterations_cnt=frame_encoder_manager.iterations_counter,
        mac_decoded_pixel=frame_encoder.get_total_mac_per_pixel(),
        lmbda=frame_encoder_manager.lmbda,
    )

    # 3. Restore training mode ---------------------------------------------- #
    frame_encoder.set_to_train()

    return encoder_logs
