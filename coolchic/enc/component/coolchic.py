# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import inspect
import math
import typing
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, TypedDict

from torch import nn, Tensor

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from enc.component.core.arm import (
    Arm,
    _get_neighbor,
    _get_non_zero_pixel_ctx_index,
    _laplace_cdf,
)
from enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
    quantize,
)
from enc.component.core.synthesis import Synthesis
from enc.component.core.upsampling import Upsampling
from enc.utils.misc import (
    MAX_ARM_MASK_SIZE,
    POSSIBLE_DEVICE,
    DescriptorCoolChic,
    DescriptorNN,
    measure_expgolomb_rate,
)


"""A cool-chic encoder is composed of:
    - A set of 2d hierarchical latent grids
    - An auto-regressive probability module
    - An upsampling module
    - A synthesis.

At its core, it is a tool to compress any spatially organized signal, with
one or more features by representing it as a set of 2d entropy coding-friendly
latent grids. After upsampling, these latent grids allows to synthesize the
desired signal.
"""

"""Dataclass to store the parameters of CoolChicEncoder for one frame."""


@dataclass
class CoolChicEncoderParameter:
    """Dataclass storing the parameters of a ``CoolChicEncoder``.

    Args:
        img_size (Tuple[int, int]): Height and width :math:`(H, W)` of the frame
            to be coded
        layers_synthesis (List[str]): Describes the architecture of the
            synthesis transform. See the :doc:`synthesis documentation
            <core/synthesis>` for more information.
        n_ft_per_res (List[int]): Number of latent features for each latent
            resolution *i.e.* ``n_ft_per_res[i]`` gives the number of channel
            :math:`C_i` of the latent with resolution :math:`\\frac{H}{2^i},
            \\frac{W}{2^i}`.
        dim_arm (int, Optional): Number of context pixels for the ARM. Also
            corresponds to the ARM hidden layer width. See the :doc:`ARM
            documentation <core/arm>` for more information. Defaults to 24
        n_hidden_layers_arm (int, Optional): Number of hidden layers in the
            ARM. Set ``n_hidden_layers_arm = 0`` for a linear ARM. Defaults
            to 2.
        upsampling_kernel_size (int, Optional): Kernel size for the upsampler.
            See the :doc:`upsampling documentation <core/upsampling>` for more
            information. Defaults to 8.
        static_upsampling_kernel (bool, Optional): Set this flag to ``True`` to
            prevent learning the upsampling kernel. Defaults to ``False``.
        encoder_gain (int, Optional): Multiply the latent by this value before
            quantization. See the documentation of Cool-chic forward pass.
            Defaults to 16.
    """
    layers_synthesis: List[str]
    n_ft_per_res: List[int]
    dim_arm: int = 24
    n_hidden_layers_arm: int = 2
    upsampling_kernel_size: int = 8
    static_upsampling_kernel: bool = False
    encoder_gain: int = 16

    # ==================== Not set by the init function ===================== #
    #: Automatically computed, number of different latent resolutions
    latent_n_grids: int = field(init=False)
    #: Height and width :math:`(H, W)` of the frame to be coded. Must be
    #: set using the ``set_image_size()`` function.
    img_size: Optional[Tuple[int, int]] = field(init=False, default=None)
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        self.latent_n_grids = len(self.n_ft_per_res)

    def set_image_size(self, img_size: Tuple[int, int]) -> None:
        """Register the field self.img_size.

        Args:
            img_size: Height and width :math:`(H, W)` of the frame to be coded
        """
        self.img_size = img_size

    def pretty_string(self) -> str:
        """Return a pretty string formatting the data within the class"""
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80

        s = "CoolChicEncoderParameter value:\n"
        s += "-------------------------------\n"
        for k in fields(self):
            s += f"{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n"
        s += "\n"
        return s



class CoolChicEncoderOutput(TypedDict):
    """``TypedDict`` representing the output of CoolChicEncoder forward.

    .. hint::

        ``torch.jit`` requires I/O of modules to be either ``Tensor``, ``List``
        or ``Dict``. So we don't use a python dataclass here and rely on
        ``TypedDict`` instead.

    Args:
        raw_out (Tensor): Output of the synthesis :math:`([B, C, H, W])`.
        rate (Tensor): rate associated to each latent (in bits). Shape is
            :math:`(N)`, with :math:`N` the total number of latent variables.
        additional_data (Dict[str, Any]): Any other data required to compute
            some logs, stored inside a dictionary
    """

    raw_out: Tensor
    rate: Tensor
    additional_data: Dict[str, Any]


class CoolChicEncoder(nn.Module):
    """CoolChicEncoder for a single frame."""

    def __init__(self, param: CoolChicEncoderParameter):
        """Instantiate a cool-chic encoder for one frame.

        Args:
            param (CoolChicEncoderParameter): Architecture of the
                `CoolChicEncoder`. See the documentation of
                `CoolChicEncoderParameter` for more information
        """
        super().__init__()

        # ? What is this for?
        torch.set_printoptions(threshold=10000000)

        # Everything is stored inside param
        self.param = param

        assert self.param.img_size is not None, (
            "You are trying to instantiate a CoolChicEncoder from a "
            "CoolChicEncoderParameter with a field img_size set to None. Use "
            "the function coolchic_encoder_param.set_img_size((H, W)) before "
            "instantiating the CoolChicEncoder."
        )

        # ================== Synthesis related stuff ================= #
        # Encoder-side latent gain applied prior to quantization, one per feature
        self.encoder_gains = param.encoder_gain

        # Populate the successive grids
        self.size_per_latent = []
        self.latent_grids = nn.ParameterList()
        for i in range(self.param.latent_n_grids):
            h_grid, w_grid = [int(math.ceil(x / (2**i))) for x in self.param.img_size]
            c_grid = self.param.n_ft_per_res[i]
            cur_size = (1, c_grid, h_grid, w_grid)

            self.size_per_latent.append(cur_size)

            # Instantiate empty tensor, we fill them later on with the function
            # self.initialize_latent_grids()
            self.latent_grids.append(
                nn.Parameter(torch.empty(cur_size), requires_grad=True)
            )

        self.initialize_latent_grids()

        # Instantiate the synthesis MLP with as many inputs as the number
        # of latent channels
        self.synthesis = Synthesis(
            sum([latent_size[1] for latent_size in self.size_per_latent]),
            self.param.layers_synthesis,
        )
        # ================== Synthesis related stuff ================= #

        # ===================== Upsampling stuff ===================== #
        self.upsampling = Upsampling(
            self.param.upsampling_kernel_size, self.param.static_upsampling_kernel
        )
        # ===================== Upsampling stuff ===================== #

        # ===================== ARM related stuff ==================== #
        # Create the probability model for the main INR. It uses a spatial context
        # parameterized by the spatial context

        # For a given mask size N (odd number e.g. 3, 5, 7), we have at most
        # (N * N - 1) / 2 context pixels in it.
        # Example, a 9x9 mask as below has 40 context pixel (indicated with 1s)
        # available to predict the pixel '*'
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 * 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0 0 0

        # No more than 40 context pixels i.e. a 9x9 mask size (see example above)
        max_mask_size = MAX_ARM_MASK_SIZE
        max_context_pixel = int((max_mask_size**2 - 1) / 2)
        assert self.param.dim_arm <= max_context_pixel, (
            f"You can not have more context pixels "
            f" than {max_context_pixel}. Found {self.param.dim_arm}"
        )

        # Mask of size 2N + 1 when we have N rows & columns of context.
        self.mask_size = max_mask_size

        # 1D tensor containing the indices of the selected context pixels.
        # register_buffer for automatic device management. We set persistent to false
        # to simply use the "automatically move to device" function, without
        # considering non_zero_pixel_ctx_index as a parameters (i.e. returned
        # by self.parameters())
        self.register_buffer(
            "non_zero_pixel_ctx_index",
            _get_non_zero_pixel_ctx_index(self.param.dim_arm),
            persistent=False,
        )

        self.arm = Arm(self.param.dim_arm, self.param.n_hidden_layers_arm)
        # ===================== ARM related stuff ==================== #

        # ======================== Monitoring ======================== #
        # Pretty string representing the decoder complexity
        self.flops_str = ""
        # Total number of multiplications to decode the image
        self.total_flops = 0.0
        # Fill the two attributes aboves
        self.get_flops()
        # ======================== Monitoring ======================== #

        # Something like ['arm', 'synthesis', 'upsampling']
        self.modules_to_send = [tmp.name for tmp in fields(DescriptorCoolChic)]

        # Track the quantization step of each neural network, None if the
        # module is not yet quantized
        self.nn_q_step: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

        # Track the exponent of the exp-golomb code used for the NN parameters.
        # None if module is not yet quantized
        self.nn_expgol_cnt: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

        # Copy of the full precision parameters, set just before calling the
        # quantize_model() function. This is done through the
        # self._store_full_precision_param() function
        self.full_precision_param = None

    # ------- Actual forward
    def forward(
        self,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: Optional[float] = 0.3,
        noise_parameter: Optional[float] = 1.0,
        AC_MAX_VAL: int = -1,
        flag_additional_outputs: bool = False,
    ) -> CoolChicEncoderOutput:
        """Perform CoolChicEncoder forward pass, to be used during the training.
        The main step are as follows:

            1. **Scale & quantize the encoder-side latent** :math:`\\mathbf{y}` to
               get the decoder-side latent

                .. math::

                    \\hat{\\mathbf{y}} = \\mathrm{Q}(\\Gamma_{enc}\\ \\mathbf{y}),

                with :math:`\\Gamma_{enc} \\in \\mathbb{R}` a scalar encoder gain
                defined in ``self.param.encoder_gains`` and :math:`\\mathrm{Q}`
                the :doc:`quantization operation <core/quantizer>`.

            2. **Measure the rate** of the decoder-side latent with the
               :doc:`ARM <core/arm>`:

                .. math::

                    \\mathrm{R}(\\hat{\\mathbf{y}}) = -\\log_2 p_{\\psi}(\\hat{\\mathbf{y}}),

               where :math:`p_{\\psi}`
               is given by the :doc:`Auto-Regressive Module (ARM) <core/arm>`.

            3. **Upsample and synthesize** the latent to get the output

                .. math::

                    \\hat{\\mathbf{x}} = f_{\\theta}(f_{\\upsilon}(\\hat{\\mathbf{y}})),

               with :math:`f_{\\psi}` the :doc:`Upsampling <core/upsampling>`
               and :math:`f_{\\theta}` the :doc:`Synthesis <core/synthesis>`.

        Args:
            quantizer_noise_type: Defaults to ``"kumaraswamy"``.
            quantizer_type: Defaults to ``"softround"``.
            soft_round_temperature: Soft round temperature.
                This is used for softround modes as well as the
                ste mode to simulate the derivative in the backward.
                Defaults to 0.3.
            noise_parameter: noise distribution parameter. Defaults to 1.0.
            AC_MAX_VAL: If different from -1, clamp the value to be in
                :math:`[-AC\\_MAX\\_VAL; AC\\_MAX\\_VAL + 1]` to write the actual bitstream.
                Defaults to -1.
            flag_additional_outputs: True to fill
                ``CoolChicEncoderOutput['additional_data']`` with many different
                quantities which can be used to analyze Cool-chic behavior.
                Defaults to False.

        Returns:
            Output of Cool-chic training forward pass.
        """

        # ! Order of the operations are important as these are asynchronous
        # ! CUDA operations. Some ordering are faster than other...

        # ------ Encoder-side: quantize the latent
        # Convert the N [1, C, H_i, W_i] 4d latents with different resolutions
        # to a single flat vector. This allows to call the quantization
        # only once, which is faster
        encoder_side_flat_latent = torch.cat(
            [latent_i.view(-1) for latent_i in self.latent_grids]
        )

        flat_decoder_side_latent = quantize(
            encoder_side_flat_latent * self.encoder_gains,
            quantizer_noise_type if self.training else "none",
            quantizer_type if self.training else "hardround",
            soft_round_temperature,
            noise_parameter,
        )

        # Clamp latent if we need to write a bitstream
        if AC_MAX_VAL != -1:
            flat_decoder_side_latent = torch.clamp(
                flat_decoder_side_latent, -AC_MAX_VAL, AC_MAX_VAL + 1
            )

        # Convert back the 1d tensor to a list of N [1, C, H_i, W_i] 4d latents.
        # This require a few additional information about each individual
        # latent dimension, stored in self.size_per_latent
        decoder_side_latent = []
        cnt = 0
        for latent_size in self.size_per_latent:
            b, c, h, w = latent_size  # b should be one
            latent_numel = b * c * h * w
            decoder_side_latent.append(
                flat_decoder_side_latent[cnt : cnt + latent_numel].view(latent_size)
            )
            cnt += latent_numel

        # ----- ARM to estimate the distribution and the rate of each latent
        # As for the quantization, we flatten all the latent and their context
        # so that the ARM network is only called once.
        # flat_latent: [N, 1] tensor describing N latents
        # flat_context: [N, context_size] tensor describing each latent context

        # Get all the context as a single 2D vector of size [B, context size]
        flat_context = torch.cat(
            [
                _get_neighbor(spatial_latent_i, self.mask_size, self.non_zero_pixel_ctx_index)
                for spatial_latent_i in decoder_side_latent
            ],
            dim=0,
        )

        # Get all the B latent variables as a single one dimensional vector
        flat_latent = torch.cat(
            [spatial_latent_i.view(-1) for spatial_latent_i in decoder_side_latent],
            dim=0
        )

        # Feed the spatial context to the arm MLP and get mu and scale
        flat_mu, flat_scale, flat_log_scale = self.arm(flat_context)

        # Compute the rate (i.e. the entropy of flat latent knowing mu and scale)
        proba = torch.clamp_min(
            _laplace_cdf(flat_latent + 0.5, flat_mu, flat_scale)
            - _laplace_cdf(flat_latent - 0.5, flat_mu, flat_scale),
            min=2**-16,  # No value can cost more than 16 bits.
        )
        flat_rate = -torch.log2(proba)

        # Upsampling and synthesis to get the output
        synthesis_output = self.synthesis(self.upsampling(decoder_side_latent))

        additional_data = {}
        if flag_additional_outputs:
            # Prepare list to accommodate the visualisations
            additional_data["detailed_sent_latent"] = []
            additional_data["detailed_mu"] = []
            additional_data["detailed_scale"] = []
            additional_data["detailed_log_scale"] = []
            additional_data["detailed_rate_bit"] = []
            additional_data["detailed_centered_latent"] = []

            # "Pointer" for the reading of the 1D scale, mu and rate
            cnt = 0
            # for i, _ in enumerate(filtered_latent):
            for index_latent_res, _ in enumerate(self.latent_grids):
                c_i, h_i, w_i = decoder_side_latent[index_latent_res].size()[-3:]
                additional_data["detailed_sent_latent"].append(
                    decoder_side_latent[index_latent_res].view((1, c_i, h_i, w_i))
                )

                # Scale, mu and rate are 1D tensors where the N latent grids
                # are flattened together. As such we have to read the appropriate
                # number of values in this 1D vector to reconstruct the i-th grid in 2D
                mu_i, scale_i, log_scale_i, rate_i = [
                    # Read h_i * w_i values starting from cnt
                    tmp[cnt : cnt + (c_i * h_i * w_i)].view((1, c_i, h_i, w_i))
                    for tmp in [flat_mu, flat_scale, flat_log_scale, flat_rate]
                ]

                cnt += c_i * h_i * w_i
                additional_data["detailed_mu"].append(mu_i)
                additional_data["detailed_scale"].append(scale_i)
                additional_data["detailed_log_scale"].append(log_scale_i)
                additional_data["detailed_rate_bit"].append(rate_i)
                additional_data["detailed_centered_latent"].append(
                    additional_data["detailed_sent_latent"][-1] - mu_i
                )

        res: CoolChicEncoderOutput = {
            "raw_out": synthesis_output,
            "rate": flat_rate,
            "additional_data": additional_data,
        }

        return res

    # ------- Getter / Setter and Initializer
    def get_param(self) -> OrderedDict[str, Tensor]:
        """Return **a copy** of the weights and biases inside the module.

        Returns:
            OrderedDict[str, Tensor]: A copy of all weights & biases in the module.
        """
        param = OrderedDict({})
        param.update(
            {
                # Detach & clone to create a copy
                f"latent_grids.{k}": v.detach().clone()
                for k, v in self.latent_grids.named_parameters()
            }
        )
        param.update({f"arm.{k}": v for k, v in self.arm.get_param().items()})
        param.update(
            {f"upsampling.{k}": v for k, v in self.upsampling.get_param().items()}
        )
        param.update(
            {f"synthesis.{k}": v for k, v in self.synthesis.get_param().items()}
        )
        return param

    def set_param(self, param: OrderedDict[str, Tensor]):
        """Replace the current parameters of the module with param.

        Args:
            param (OrderedDict[str, Tensor]): Parameters to be set.
        """
        self.load_state_dict(param)

    def initialize_latent_grids(self) -> None:
        """Initialize the latent grids. The different tensors composing
        the latent grids must have already been created e.g. through
        ``torch.empty()``.
        """
        for latent_index, latent_value in enumerate(self.latent_grids):
            self.latent_grids[latent_index] = nn.Parameter(
                torch.zeros_like(latent_value), requires_grad=True
            )

    def reinitialize_parameters(self):
        """Reinitialize in place the different parameters of a CoolChicEncoder
        namely the latent grids, the arm, the upsampling and the weights.
        """
        self.arm.reinitialize_parameters()
        self.upsampling.reinitialize_parameters()
        self.synthesis.reinitialize_parameters()
        self.initialize_latent_grids()

        # Reset the quantization steps and exp-golomb count of the neural
        # network to None since we are resetting the parameters.
        self.nn_q_step: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }
        self.nn_expgol_cnt: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

    def _store_full_precision_param(self) -> None:
        """Store the current parameters inside self.full_precision_param

        This function checks that there is no self.nn_q_step and
        self.nn_expgol_cnt already saved. This would mean that we no longer
        have full precision parameters but quantized ones.
        """

        if self.full_precision_param is not None:
            print(
                "Warning: overwriting already saved full-precision parameters"
                " in CoolChicEncoder _store_full_precision_param()."
            )

        # Check that we haven't already quantized the network by looking at
        # the nn_expgol_cnt and nn_q_step dictionaries
        no_q_step = True
        for _, q_step_dict in self.nn_q_step.items():
            for _, q_step in q_step_dict.items():
                if q_step is not None:
                    no_q_step = False
        assert no_q_step, (
            "Trying to store full precision parameters, while CoolChicEncoder "
            "nn_q_step attributes is not full of None. This means that the "
            "parameters have already been quantized... aborting!"
        )

        no_expgol_cnt = True
        for _, expgol_cnt_dict in self.nn_expgol_cnt.items():
            for _, expgol_cnt in expgol_cnt_dict.items():
                if expgol_cnt is not None:
                    no_expgol_cnt = False
        assert no_expgol_cnt, (
            "Trying to store full precision parameters, while CoolChicEncoder "
            "nn_expgol_cnt attributes is not full of None. This means that the "
            "parameters have already been quantized... aborting!"
        )

        # All good, simply save the parameters
        self.full_precision_param = self.get_param()

    def _load_full_precision_param(self) -> None:
        assert self.full_precision_param is not None, (
            "Trying to load full precision parameters but "
            "self.full_precision_param is None"
        )

        self.set_param(self.full_precision_param)

        # Reset the side information about the quantization step and expgol cnt
        # so that the rate is no longer computed by the test() function.
        self.nn_q_step: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

        self.nn_expgol_cnt: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }



    # ------- Get flops, neural network rates and quantization step
    def get_flops(self) -> None:
        """Compute the number of MAC & parameters for the model.
        Update ``self.total_flops`` (integer describing the number of total MAC)
        and ``self.flops_str``, a pretty string allowing to print the model
        complexity somewhere.

        .. attention::

            ``fvcore`` measures MAC (multiplication & accumulation) but calls it
            FLOP (floating point operation)... We do the same here and call
            everything FLOP even though it would be more accurate to use MAC.
        """
        # print("Ignoring get_flops")
        # Count the number of floating point operations here. It must be done before
        # torch scripting the different modules.
        flops = FlopCountAnalysis(
            self,
            (
                "none",  # Quantization noise
                "hardround",  # Quantizer type
                0.3,  # Soft round temperature
                0.1,  # Noise parameter
                -1,  # AC_MAX_VAL
                False,  # Flag additional outputs
            ),
        )
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)

        self.total_flops = flops.total()
        self.flops_str = flop_count_table(flops)
        del flops

    def get_network_rate(self) -> DescriptorCoolChic:
        """Return the rate (in bits) associated to the parameters
        (weights and biases) of the different modules

        Returns:
            DescriptorCoolChic: The rate (in bits) associated with the weights
            and biases of each module
        """
        rate_per_module: DescriptorCoolChic = {
            module_name: {"weight": 0.0, "bias": 0.0}
            for module_name in self.modules_to_send
        }

        for module_name in self.modules_to_send:
            cur_module = getattr(self, module_name)
            rate_per_module[module_name] = measure_expgolomb_rate(
                cur_module,
                self.nn_q_step.get(module_name),
                self.nn_expgol_cnt.get(module_name),
            )

        return rate_per_module

    def get_network_quantization_step(self) -> DescriptorCoolChic:
        """Return the quantization step associated to the parameters (weights
        and biases) of the different modules. Those quantization can be
        ``None`` if the model has not yet been quantized.

        Returns:
            DescriptorCoolChic: The quantization step associated with the
            weights and biases of each module.
        """
        return self.nn_q_step

    def get_network_expgol_count(self) -> DescriptorCoolChic:
        """Return the Exp-Golomb count parameter associated to the parameters
        (weights and biases) of the different modules. Those quantization can be
        ``None`` if the model has not yet been quantized.

        Returns:
            DescriptorCoolChic: The Exp-Golomb count parameter associated
            with the weights and biases of each module.
        """
        return self.nn_expgol_cnt


    def str_complexity(self) -> str:
        """Return a string describing the number of MAC (**not mac per pixel**) and the
        number of parameters for the different modules of CoolChic

        Returns:
            str: A pretty string about CoolChic complexity.
        """

        if not self.flops_str:
            self.get_flops()

        msg_total_mac = "----------------------------------\n"
        msg_total_mac += (
            f"Total MAC / decoded pixel: {self.get_total_mac_per_pixel():.1f}"
        )
        msg_total_mac += "\n----------------------------------"

        return self.flops_str + "\n\n" + msg_total_mac

    def get_total_mac_per_pixel(self) -> float:
        """Count the number of Multiplication-Accumulation (MAC) per decoded pixel
        for this model.

        Returns:
            float: number of floating point operations per decoded pixel.
        """

        if not self.flops_str:
            self.get_flops()

        n_pixels = self.param.img_size[-2] * self.param.img_size[-1]
        return self.total_flops / n_pixels

    # ------- Useful functions
    def to_device(self, device: POSSIBLE_DEVICE) -> None:
        """Push a model to a given device.

        Args:
            device (POSSIBLE_DEVICE): The device on which the model should run.
        """

        assert device in typing.get_args(
            POSSIBLE_DEVICE
        ), f"Unknown device {device}, should be in {typing.get_args(POSSIBLE_DEVICE)}"
        self = self.to(device)

        # Push integerized weights and biases of the mlp (resp qw and qb) to
        # the required device
        for idx_layer, layer in enumerate(self.arm.mlp):
            if hasattr(layer, "qw"):
                if layer.qw is not None:
                    self.arm.mlp[idx_layer].qw = layer.qw.to(device)

            if hasattr(layer, "qb"):
                if layer.qb is not None:
                    self.arm.mlp[idx_layer].qb = layer.qb.to(device)
