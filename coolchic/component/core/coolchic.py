# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, OrderedDict, Tuple, TypedDict

import torch
import torch.nn.functional as F
from einops import rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch import Tensor, nn

from coolchic.component.core.arm import (
    Arm,
    Ifce,
    _get_mask_size_ctx,
    _get_neighbor,
    _get_non_zero_pixel_ctx_index,
    _laplace_cdf,
)
from coolchic.component.core.noise import CommonGaussianNoiseGenerator
from coolchic.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
    quantize,
)
from coolchic.component.core.synthesis import Synthesis
from coolchic.component.core.types import DescriptorCoolChic, DescriptorNN
from coolchic.component.core.upsampling import Upsampling, fixed_upsampling

"""A cool-chic encoder is composed of:
    - A set of 2d hierarchical latent grids
    - An auto-regressive probability module + Inter-feature context extractor.
    - An upsampling module
    - A synthesis.

At its core, it is a tool to compress any spatially organized signal, with
one or more features by representing it as a set of 2d entropy coding-friendly
latent grids. After upsampling, these latent grids allows to synthesize the
desired signal.
"""


@dataclass
class CoolChicEncoderParameter:
    """Dataclass storing the parameters of a ``CoolChicEncoder``.

    Args:
        layers_synthesis (List[str]): Describes the architecture of the synthesis transform.
            See the :doc:`synthesis documentation <synthesis>` for more information.
        linear_stabiliser_synth (bool): Flag indicating the usage of the linear stabiliser
            for the synthesis.
        ups_k_size (int): Upsampling kernel size for the transposed
            convolutions. See the :doc:`upsampling documentation <upsampling>`
            for more information.
        ups_preconcat_k_size (int): Upsampling kernel size for the
            pre-concatenation convolutions. See the
            :doc:`upsampling documentation <upsampling>` for more
            information.
        ifce_resolution (Optional[Tuple[int, int]]): Lowest and highest base two downsampling
            of the latent using the IFCEs. E.g., (0, 2) means latents between downsampling 1/2^0
            and 1/2^2. Set to None to disable.
        output_feature_ifce (int): Number of output features of the IFCEs. Ignored if
            ifce_resolution is None.
        spatial_context_arm (int): Number of spatial contexts for the ARM.
        linear_stabiliser_arm (bool): Flag indicating the usage of the linear stabiliser for the ARM
        n_hidden_layers_arm (int): Number of hidden layers in the ARM. Set to zero for a linear ARM.
        latent_resolution (Tuple[int, int]): Lowest and highest base two downsampling
            of the latent grids. E.g., (0, 4) means 5 latent grids from downsampling 1/2^0
            to 1/2^4.
        hyper_latent_resolution (Optional[Tuple[int, int]]): Identical to latent_resolution but for
            hyperlatent *i.e.,* additional latent grids which are used only for the entropy
            modeling and not by the synthesis. Set to None to disable
        flag_common_randomness (bool). Flag indicating the usage of common randomness latent grids,
            with resolution identical to the latent_resolution parameters.
        img_size (Tuple[int, int]): Height and width :math:`(H, W)` of the frame
            to be coded
        encoder_gain (int): Multiply the latent by this value before quantization. Defaults to 16.
        final_upsampling_type (Literal["nearest", "bilinear", "bicubic"]). If the resolution of
            the biggest latent grid is smaller than the input image, upsample it using the
            specified filter to the image size.
    """

    # ---- Synthesis
    layers_synthesis: List[str]
    linear_stabiliser_synth: bool
    input_feature_synthesis: int = field(init=False)

    # ---- Upsampling
    ups_k_size: int
    ups_preconcat_k_size: int

    # ---- Entropy model
    ifce_resolution: Optional[Tuple[int, int]]
    output_feature_ifce: int

    spatial_context_arm: int
    linear_stabiliser_arm: bool
    n_hidden_layers_arm: int
    total_context_arm: int = field(init=False)

    # ---- Latent grids and hyper latent grids
    latent_resolution: Tuple[int, int]
    hyperlatent_resolution: Optional[Tuple[int, int]]
    flag_common_randomness: bool

    # ---- Others
    img_size: Tuple[int, int]
    # If the synthesis output is smaller than img_size i.e. when the highest latent resolution
    # is not 1/1, there is a final upsampling.
    final_upsampling_type: Literal["nearest", "bilinear", "bicubic"]
    encoder_gain: int = 16

    # ==================== Not set by the init function ===================== #
    # Set to true if there is at least one feature of common randomness requested
    flag_ifce: bool = field(init=False)
    flag_hyperlatent: bool = field(init=False)
    cr_latent_resolution: Optional[Tuple[int, int]] = field(init=False)

    # size_per_latent[i] = Dimension of the i-th latent: (1, 1, H_i, W_i)
    # size_per_latent[0] is the biggest
    size_per_latent: List[Tuple[int, int, int, int]] = field(init=False, default_factory=lambda: [])

    # Same thing but for the common randomness latent
    # size_per_latent_cr[i] = Dimension of the i-th common randomness latent: (1, 1, H_i, W_i)
    # size_per_latent_cr[0] is the biggest
    size_per_latent_cr: List[Tuple[int, int, int, int]] = field(
        init=False, default_factory=lambda: []
    )

    # flag_is_hyperlatent[i] = True --> i-th latent is only used for the entropy
    # decoding and discarded before the upsampling/synthesis
    flag_is_hyperlatent: List[bool] = field(init=False, default_factory=lambda: [])

    # input_features_ifce[i] = number of input feature for the ifce associated to the
    # i-th latent grid. Set to zero if the i-th latent does not have an IFCE associated
    input_features_ifce: List[int] = field(init=False, default_factory=lambda: [])
    # Total number of latent transmitted
    n_latent_grids: int = field(init=False)
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):

        # Order is important. Some parameters set in post_init_latent() are reused
        # in the initialization of the synthesis or IFCE for instance.
        self.post_init_latent()
        self.post_init_arm()
        self.post_init_common_randomness()
        self.post_init_synthesis()
        self.post_init_ifce()

    def post_init_latent(self) -> None:
        self.flag_hyperlatent = self.hyperlatent_resolution is not None

        # Compute all latent spatial dimension and fill the flag_is_hyperlatent list to indicate
        # which grids are to be discarded before the synthesis
        if self.flag_hyperlatent:
            min_downsampling = min(self.latent_resolution + self.hyperlatent_resolution)
            max_downsampling = max(self.latent_resolution + self.hyperlatent_resolution)
        else:
            min_downsampling, max_downsampling = self.latent_resolution

        for i in range(min_downsampling, max_downsampling + 1):
            h_grid, w_grid = [int(math.ceil(x / (2**i))) for x in self.img_size]
            cur_size = (1, 1, h_grid, w_grid)

            # Add the grid if it falls inside the required latent resolution
            if self.latent_resolution[0] <= i <= self.latent_resolution[1]:
                self.size_per_latent.append(cur_size)
                self.flag_is_hyperlatent.append(False)

            if self.flag_hyperlatent:
                # Add the grid if it falls inside the required hyperlatent resolution
                if self.hyperlatent_resolution[0] <= i <= self.hyperlatent_resolution[1]:
                    self.size_per_latent.append(cur_size)
                    self.flag_is_hyperlatent.append(True)

        self.n_latent_grids = len(self.size_per_latent)

        if self.flag_common_randomness:
            for i in range(self.latent_resolution[0], self.latent_resolution[1] + 1):
                h_grid, w_grid = [int(math.ceil(x / (2**i))) for x in self.img_size]
                cur_size = (1, 1, h_grid, w_grid)
                self.size_per_latent_cr.append(cur_size)

    def post_init_arm(self) -> None:
        self.total_context_arm = self.spatial_context_arm + self.output_feature_ifce

    def post_init_common_randomness(self) -> None:
        if self.flag_common_randomness:
            # Common randomness has the same resolution than the latent variables
            self.cr_latent_resolution = (self.latent_resolution[0], self.latent_resolution[1])
        else:
            self.cr_latent_resolution = None

    def post_init_synthesis(self) -> None:
        # latent_resolution = (0, 6) --> 7 = 6 - 0 + 1 latent features
        self.input_feature_synthesis = self.latent_resolution[1] - self.latent_resolution[0] + 1
        if self.flag_common_randomness:
            self.input_feature_synthesis *= 2

    def post_init_ifce(self) -> None:
        self.flag_ifce = self.ifce_resolution is not None

        for i, size_latent_i in enumerate(self.size_per_latent):
            # We assume identical downsampling ratio for height and width
            downsampling_ratio = int(math.ceil(math.log2(self.img_size[0] / size_latent_i[-2])))

            if not self.flag_ifce:
                self.input_features_ifce.append(0)

            # We do have an IFCE
            elif self.ifce_resolution[0] <= downsampling_ratio <= self.ifce_resolution[1]:
                # How many latents are already decoded when we're decoding latent_i
                # max(X, 1) because we always have at least one input feature. Padding if need be
                self.input_features_ifce.append(max(self.n_latent_grids - 1 - i, 1))
            else:
                self.input_features_ifce.append(0)

    def pretty_string(self) -> str:
        """Return a pretty string presenting the CoolChicEncoderParameter."""
        ATTRIBUTE_WIDTH = 35
        VALUE_WIDTH = 80

        s = ""
        for k in fields(self):
            v = getattr(self, k.name)

            # Print only height and width
            if k.name.startswith("size_per_latent"):
                v = [v_i[-2:] for v_i in v]

            s += f"{k.name:<{ATTRIBUTE_WIDTH}}: {str(v):<{VALUE_WIDTH}}\n"
        s += "\n"
        return s


class CoolChicEncoderOutput(TypedDict):
    """``TypedDict`` representing the output of CoolChicEncoder forward.

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

        # Everything is stored inside param
        self.param = param

        assert self.param.img_size is not None, (
            "You are trying to instantiate a CoolChicEncoder from a "
            "CoolChicEncoderParameter with a field img_size set to None. Use "
            "the function coolchic_encoder_param.set_img_size((H, W)) before "
            "instantiating the CoolChicEncoder."
        )

        # ================== Latent related stuff ================= #
        # Encoder-side latent gain applied prior to quantization, one per feature
        self.encoder_gains = param.encoder_gain

        self.latent_grids = instantiate_latent_grids_from_cc_param(self.param)
        self.initialize_latent_grids()

        self.cr = instantiate_common_randomness_from_cc_param(self.param)
        # ================== Latent related stuff ================= #

        # ===================== ARM related stuff ==================== #
        # All context pixels are centered in a mask_size x mask_size window centered
        # on the pixel to be entropy coded
        self.mask_size = _get_mask_size_ctx(self.param.spatial_context_arm)

        # 1D tensor containing the indices of the selected context pixels.
        # register_buffer for automatic device management. We set persistent to false
        # to simply use the "automatically move to device" function, without
        # considering non_zero_pixel_ctx_index as a parameters (i.e. returned
        # by self.parameters())
        self.register_buffer(
            "non_zero_pixel_ctx_index",
            _get_non_zero_pixel_ctx_index(self.param.spatial_context_arm),
            persistent=False,
        )

        self.arm = instantiate_arm_from_cc_param(self.param)
        self.synthesis = instantiate_syn_from_cc_param(self.param)
        self.upsampling = instantiate_ups_from_cc_param(self.param)
        self.ifce = instantiate_ifce_from_cc_param(self.param)
        # ===================== ARM related stuff ==================== #

        # Something like ['arm', 'synthesis', 'upsampling', 'ifce']
        self.modules_to_send = [tmp.name for tmp in fields(DescriptorCoolChic)]

        if self.ifce is None:
            self.modules_to_send.remove("ifce")

        # ======================== Monitoring ======================== #
        # Pretty string representing the decoder complexity
        self.flops_str = ""
        # Total number of multiplications to decode the image
        self.total_flops = 0.0
        self.flops_per_module = {k: 0 for k in self.modules_to_send}
        # Fill the two attributes aboves
        self.get_flops()
        # ======================== Monitoring ======================== #

        # Track the quantization step of each neural network, None if the
        # module is not yet quantized. Default initialization of DescriptorCoolChic
        # is None everywhere
        self.nn_q_step = DescriptorCoolChic()

        # Track the exponent of the exp-golomb code used for the NN parameters.
        # None if module is not yet quantized. Default initialization of DescriptorCoolChic
        # is None everywhere
        self.nn_expgol_cnt = DescriptorCoolChic()
        # Copy of the full precision parameters, set just before calling the
        # quantize_model() function. This is done through the
        # self._store_full_precision_param() function
        self.full_precision_param = None

    # ------- Actual forward
    def forward(
        self,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "gaussian",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: Optional[Tensor] = torch.tensor(0.35),
        noise_parameter: Optional[Tensor] = torch.tensor(0.22),
        AC_MAX_VAL: int = -1,
        flag_additional_outputs: bool = False,
        no_common_randomness: bool = False,
        only_common_randomness: bool = False,
    ) -> CoolChicEncoderOutput:
        """Perform CoolChicEncoder forward pass, to be used during the training.
        The main step are as follows:

            1. **Scale & quantize the encoder-side latent** :math:`\\mathbf{y}` to
               get the decoder-side latent

                .. math::

                    \\hat{\\mathbf{y}} = \\mathrm{Q}(\\Gamma_{enc}\\ \\mathbf{y}),

                with :math:`\\Gamma_{enc} \\in \\mathbb{R}` a scalar encoder gain
                defined in ``self.param.encoder_gains`` and :math:`\\mathrm{Q}`
                the :doc:`quantization operation <quantizer>`.

            2. **Measure the rate** of the decoder-side latent with the :doc:`ARM and IFCE <arm>`:

                .. math::

                    \\mathrm{R}(\\hat{\\mathbf{y}}) = -\\log_2 p_{\\psi}(\\hat{\\mathbf{y}}),

               where :math:`p_{\\psi}` is given by the :doc:`Auto-Regressive Module (ARM) <arm>`.

            3. **Upsample and synthesize** the latent to get the output

                .. math::

                    \\hat{\\mathbf{x}} = f_{\\theta}(f_{\\upsilon}(\\hat{\\mathbf{y}})),

               with :math:`f_{\\psi}` the :doc:`Upsampling <upsampling>`
               and :math:`f_{\\theta}` the :doc:`Synthesis <synthesis>`.

        Args:
            quantizer_noise_type: Defaults to ``"gaussian"``.
            quantizer_type: Defaults to ``"softround"``.
            soft_round_temperature: Soft round temperature.
                This is used for softround modes as well as the
                ste mode to simulate the derivative in the backward.
                Defaults to 0.35.
            noise_parameter: noise distribution parameter. Defaults to 0.22.
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

        decoder_side_latent = self.get_quantize_latent(
            quantizer_noise_type=quantizer_noise_type,
            quantizer_type=quantizer_type,
            soft_round_temperature=soft_round_temperature,
            noise_parameter=noise_parameter,
            AC_MAX_VAL=AC_MAX_VAL,
        )

        # ----- ARM to estimate the distribution and the rate of each latent
        # As for the quantization, we flatten all the latent and their context
        # so that the ARM network is only called once.
        # flat_latent: [N, 1] tensor describing N latents
        # flat_context: [N, context_size] tensor describing each latent context

        # Get all the context as a single 2D vector of size [B, context size]
        flat_context_spatial = []
        flat_latent = []
        flat_context_inter_ft = []

        if self.param.flag_ifce:
            _, intermediate_latent_ups = fixed_upsampling(decoder_side_latent, mode="nearest")

        for idx_latent, spatial_latent_i in enumerate(decoder_side_latent):
            if spatial_latent_i.numel() == 0:
                continue

            flat_latent.append(spatial_latent_i.view(-1))
            cur_context_spatial = _get_neighbor(
                spatial_latent_i, self.non_zero_pixel_ctx_index, self.mask_size
            )
            cur_context_spatial = rearrange(cur_context_spatial, "b 1 n_context -> b n_context")
            flat_context_spatial.append(cur_context_spatial)

            if self.param.flag_ifce and self.param.input_features_ifce[idx_latent] > 0:
                already_decoded_latent = intermediate_latent_ups[
                    len(self.latent_grids) - 1 - idx_latent
                ]

                cur_context_inter_ft = self.ifce(
                    # Flatten for the ARM forward
                    rearrange(already_decoded_latent, "1 c h w -> (h w) c"),
                    idx_latent,
                )

                cur_context_inter_ft = rearrange(
                    cur_context_inter_ft,
                    "(h w) c -> 1 c h w",
                    h=already_decoded_latent.size()[2],
                    w=already_decoded_latent.size()[3],
                )

                # Interpolate one last time to reach the resolution of spatial_latent i
                h_i, w_i = spatial_latent_i.size()[-2:]
                cur_context_inter_ft = F.interpolate(
                    cur_context_inter_ft, scale_factor=2.0, mode="nearest"
                )[:, :, :h_i, :w_i]

                inter_neighbors = rearrange(cur_context_inter_ft, "b c h w -> (b h w) c")

                # Pad with zeros
                # padded_inter_neighbors = torch.zeros(h_i*w_i, self.param.n_out_ifce, device=inter_neighbors.device)
                # padded_inter_neighbors[:, :inter_neighbors.shape[1]] = inter_neighbors

                flat_context_inter_ft.append(inter_neighbors)

            # No inter feature ARM for this level
            elif self.param.flag_ifce:
                h_i, w_i = spatial_latent_i.size()[-2:]
                padded_inter_neighbors = torch.zeros(
                    h_i * w_i, self.param.output_feature_ifce, device=spatial_latent_i.device
                )
                flat_context_inter_ft.append(padded_inter_neighbors)

        flat_context_spatial = torch.cat(flat_context_spatial, dim=0)
        flat_latent = torch.cat(flat_latent, dim=0)

        if self.param.flag_ifce:
            flat_context_inter_ft = torch.cat(flat_context_inter_ft, dim=0)
            flat_context = torch.cat((flat_context_spatial, flat_context_inter_ft), dim=1)
        else:
            flat_context = flat_context_spatial

        # Feed the spatial context to the arm MLP and get mu and scale
        flat_mu, flat_scale = self.arm.reparameterize_output(self.arm(flat_context))

        # Compute the rate (i.e. the entropy of flat latent knowing mu and scale)
        proba = torch.clamp_min(
            _laplace_cdf(flat_latent + 0.5, flat_mu, flat_scale)
            - _laplace_cdf(flat_latent - 0.5, flat_mu, flat_scale),
            min=2**-16,  # No value can cost more than 16 bits.
        )
        flat_rate = -torch.log2(proba)

        # Discard the hyperlatent

        # Get only feature map assigned for the image reconstruction
        decoder_side_latent_syn = [
            x for x, m in zip(decoder_side_latent, self.param.flag_is_hyperlatent) if not m
        ]
        ups_latent = self.upsampling(decoder_side_latent_syn)

        # Upsampling and synthesis to get the output
        if self.param.flag_common_randomness:
            # ups_noise is [1, C, H, W] where C = len(self.cr) and H, W is the
            # spatial resolution of the highest resolution in self.cr e.g.
            # self.cr[0].size()[-2:].
            # If needed we interpolate once more to reach the resolution of the
            # image to be decoded.
            ups_noise, _ = fixed_upsampling(self.cr)
            ups_noise = F.interpolate(ups_noise, size=self.param.img_size, mode="bicubic")

            if no_common_randomness:
                ups_noise = ups_noise * 0

            if only_common_randomness:
                ups_latent = ups_latent * 0

            syn_in = torch.cat([ups_latent, ups_noise], dim=1)
        else:
            syn_in = ups_latent

        synth_out = self.synthesis(syn_in)

        # Upsample the output of the synthesis with a bicubic if required
        synthesis_output = F.interpolate(
            synth_out, size=self.param.img_size, mode=self.param.final_upsampling_type
        )

        # Trim out additional pixels due to the final upsampling
        synthesis_output = synthesis_output[
            :, :, : self.param.img_size[0], : self.param.img_size[1]
        ]

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
            # print(torch.cat([x for x in self.cr], dim=0).view(-1).exp().detach().cpu())
            for index_latent_res, _ in enumerate(self.latent_grids):
                c_i, h_i, w_i = decoder_side_latent[index_latent_res].size()[-3:]
                additional_data["detailed_sent_latent"].append(
                    decoder_side_latent[index_latent_res].view((1, c_i, h_i, w_i))
                )

                # Scale, mu and rate are 1D tensors where the N latent grids
                # are flattened together. As such we have to read the appropriate
                # number of values in this 1D vector to reconstruct the i-th grid in 2D
                mu_i, scale_i, rate_i = [
                    # Read h_i * w_i values starting from cnt
                    tmp[cnt : cnt + (c_i * h_i * w_i)].view((1, c_i, h_i, w_i))
                    for tmp in [flat_mu, flat_scale, flat_rate]
                ]

                cnt += c_i * h_i * w_i
                additional_data["detailed_mu"].append(mu_i)
                additional_data["detailed_scale"].append(scale_i)
                additional_data["detailed_rate_bit"].append(rate_i)
                additional_data["detailed_centered_latent"].append(
                    additional_data["detailed_sent_latent"][-1] - mu_i
                )

            additional_data["detailed_ups_latent"] = ups_latent
            additional_data["synthesis"] = synth_out
            # additional_data["fixer"] = fixer_out

            if self.param.flag_common_randomness:
                additional_data["detailed_ups_noise"] = ups_noise

        res: CoolChicEncoderOutput = {
            "raw_out": synthesis_output,
            "rate": flat_rate,
            "additional_data": additional_data,
        }

        return res

    def get_quantize_latent(
        self,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: Optional[Tensor] = torch.tensor(0.3),
        noise_parameter: Optional[Tensor] = torch.tensor(1.0),
        AC_MAX_VAL: int = -1,
    ):
        # ------ Encoder-side: quantize the latent
        # Convert the N [1, C, H_i, W_i] 4d latents with different resolutions
        # to a single flat vector. This allows to call the quantization
        # only once, which is faster
        encoder_side_flat_latent = torch.cat([latent_i.view(-1) for latent_i in self.latent_grids])

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
                flat_decoder_side_latent, -AC_MAX_VAL, AC_MAX_VAL - 1
            )

        # Convert back the 1d tensor to a list of N [1, C, H_i, W_i] 4d latents.
        # This require a few additional information about each individual
        # latent dimension, stored in self.param.size_per_latent
        decoder_side_latent = []
        cnt = 0
        for latent_size in self.param.size_per_latent:
            b, c, h, w = latent_size  # b should be one
            latent_numel = b * c * h * w
            decoder_side_latent.append(
                flat_decoder_side_latent[cnt : cnt + latent_numel].view(latent_size)
            )
            cnt += latent_numel

        return decoder_side_latent

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
        param.update({f"upsampling.{k}": v for k, v in self.upsampling.get_param().items()})
        param.update({f"synthesis.{k}": v for k, v in self.synthesis.get_param().items()})

        if self.ifce is not None:
            param.update({f"ifce.{k}": v for k, v in self.ifce.get_param().items()})

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
        # Default DescriptorCoolChic initialization is None.
        self.nn_q_step = DescriptorCoolChic()
        self.nn_expgol_cnt = DescriptorCoolChic()

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
        no_expgol_cnt = True

        for field_nn in fields(DescriptorCoolChic):
            for field_wb in fields(DescriptorNN):
                q_step = self.nn_q_step.get_value(field_nn.name, field_wb.name)
                expgol_cnt = self.nn_expgol_cnt.get_value(field_nn.name, field_wb.name)

                if q_step is not None:
                    no_q_step = False
                if expgol_cnt is not None:
                    no_expgol_cnt = False

        assert no_q_step and no_expgol_cnt, (
            "Trying to store full precision parameters, while CoolChicEncoder "
            "nn_q_step or nn_expgol_cnt attributes are not full of None. This means that the "
            "parameters have already been quantized... aborting!"
        )

        # All good, simply save the parameters
        self.full_precision_param = self.get_param()

    def _load_full_precision_param(self) -> None:
        assert self.full_precision_param is not None, (
            "Trying to load full precision parameters but self.full_precision_param is None"
        )

        self.set_param(self.full_precision_param)

        # Reset the side information about the quantization step and expgol cnt
        # so that the rate is no longer computed by the test() function.
        # Default init --> None
        self.nn_q_step = DescriptorCoolChic()
        self.nn_expgol_cnt = DescriptorCoolChic()

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

        self = self.train(mode=False)

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
        for k in self.flops_per_module:
            self.flops_per_module[k] = flops.by_module()[k]

        self.flops_str = flop_count_table(flops)
        del flops

        self = self.train(mode=True)

    def get_network_rate(self) -> Tuple[DescriptorCoolChic, int]:
        """Return the rate (in bits) associated to the parameters
        (weights and biases) of the different modules

        Returns:
            Tuple[DescriptorCoolChic, int]: The rate (in bits) associated with
            the weights and biases of each module. Also return the total rate
            in bits.
        """
        rate_per_module = DescriptorCoolChic(
            arm=DescriptorNN(weight=0.0, bias=0.0),
            ifce=DescriptorNN(weight=0.0, bias=0.0),
            upsampling=DescriptorNN(weight=0.0, bias=0.0),
            synthesis=DescriptorNN(weight=0.0, bias=0.0),
        )
        total_rate = 0.0

        for module_name in self.modules_to_send:
            cur_module = getattr(self, module_name)
            module_rate = measure_expgolomb_rate(
                cur_module,
                self.nn_q_step.get_value(module_name),
                self.nn_expgol_cnt.get_value(module_name),
            )
            rate_per_module.set_value(module_rate, module_name)

        total_rate = rate_per_module.sum()

        return rate_per_module, total_rate

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
        (weights and biases) of the different modules. Those exp-golomb param
        can be ``None`` if the model has not yet been quantized.

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
        msg_total_mac += f"Total MAC / decoded pixel: {self.get_total_mac_per_pixel():.1f}"
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
    def to_device(self, device: torch.device) -> None:
        """Push a model to a given device."""
        self = self.to(device)

        if self.param.flag_common_randomness:
            for i in range(len(self.cr)):
                self.cr[i] = self.cr[i].to(device)

        # # Push integerized weights and biases of the mlp (resp qw and qb) to
        # # the required device
        # for idx_layer, layer in enumerate(self.arm.mlp):
        #     if hasattr(layer, "qw"):
        #         if layer.qw is not None:
        #             self.arm.mlp[idx_layer].qw = layer.qw.to(device)

        #     if hasattr(layer, "qb"):
        #         if layer.qb is not None:
        #             self.arm.mlp[idx_layer].qb = layer.qb.to(device)

    def pretty_string(self) -> str:
        """Get a pretty string detailing the complexity of a ``CoolChicEncoder``

        Returns:
            str: a pretty string ready to be printed out
        """

        s = ""

        if not self.flops_str:
            self.get_flops()

        n_pixels = self.param.img_size[-2] * self.param.img_size[-1]
        total_mac_per_pix = self.get_total_mac_per_pixel()

        ups_complexity = self.flops_per_module["upsampling"] / n_pixels
        ups_share_complexity = 100 * ups_complexity / total_mac_per_pix

        arm_complexity = self.flops_per_module["arm"] / n_pixels
        arm_share_complexity = 100 * arm_complexity / total_mac_per_pix

        syn_complexity = self.flops_per_module["synthesis"] / n_pixels
        syn_share_complexity = 100 * syn_complexity / total_mac_per_pix

        s = (
            f"   - {'ARM':<14} {arm_complexity:5.0f} MAC / pixel; {arm_share_complexity:4.1f} % of the complexity\n"
            f"   - {'Upsampling':<14} {ups_complexity:5.0f} MAC / pixel; {ups_share_complexity:4.1f} % of the complexity\n"
            f"   - {'Synthesis':<14} {syn_complexity:5.0f} MAC / pixel; {syn_share_complexity:4.1f} % of the complexity\n"
        )

        if "ifce" in self.flops_per_module:
            ifce_complexity = self.flops_per_module["ifce"] / n_pixels
            ifce_share_complexity = 100 * ifce_complexity / total_mac_per_pix
            s += f"   - {'Inter ft ARM':<14} {ifce_complexity:5.0f} MAC / pixel; {ifce_share_complexity:4.1f} % of the complexity\n"

        return s


def instantiate_latent_grids_from_cc_param(param: CoolChicEncoderParameter) -> nn.ParameterList:
    return nn.ParameterList(
        [nn.Parameter(torch.empty(size_i), requires_grad=True) for size_i in param.size_per_latent]
    )


def instantiate_common_randomness_from_cc_param(param: CoolChicEncoderParameter) -> List[Tensor]:
    common_noise_generator = CommonGaussianNoiseGenerator()
    cr = [common_noise_generator.sample(size) for size in param.size_per_latent_cr]
    return cr


def instantiate_arm_from_cc_param(param: CoolChicEncoderParameter) -> Arm:
    return Arm(
        param.total_context_arm,
        param.n_hidden_layers_arm,
        flag_linear_stabiliser=param.linear_stabiliser_arm,
    )


def instantiate_syn_from_cc_param(param: CoolChicEncoderParameter) -> Synthesis:
    return Synthesis(
        param.input_feature_synthesis,
        param.layers_synthesis,
        param.linear_stabiliser_synth,
        param.flag_common_randomness,
    )


def instantiate_ups_from_cc_param(param: CoolChicEncoderParameter) -> Upsampling:
    # If latent_resolution = (1, 6), there 6 upsampling to go from a downsampling of 2**-6 to 2**0
    n_ups = param.latent_resolution[1]
    return Upsampling(
        ups_k_size=param.ups_k_size,
        ups_preconcat_k_size=param.ups_preconcat_k_size,
        n_ups_kernel=n_ups,
        n_ups_preconcat_kernel=n_ups,
    )


def instantiate_ifce_from_cc_param(param: CoolChicEncoderParameter) -> Optional[Ifce]:
    if param.flag_ifce:
        return Ifce(param.input_features_ifce, param.output_feature_ifce)
    else:
        return None


@torch.no_grad()
def measure_expgolomb_rate(
    q_module: nn.Module, q_step: DescriptorNN, expgol_cnt: DescriptorNN
) -> DescriptorNN:
    """Get the rate associated with the current parameters.

    Returns:
        DescriptorNN: The rate of the different modules wrapped inside a dictionary
            of float. It does **not** return tensor so no back propagation is possible
    """
    # Concatenate the sent parameters here to measure the entropy later
    sent_param = DescriptorNN(bias=[], weight=[])
    rate_param = DescriptorNN(bias=0.0, weight=0.0)

    param = q_module.get_param()
    # Retrieve all the sent item
    for parameter_name, parameter_value in param.items():
        if ".weight" in parameter_name:
            current_q_step = q_step.weight
        elif ".bias" in parameter_name:
            current_q_step = q_step.bias

        # Current quantization step is None because the module is not yet
        # quantized. Return an all zero rate
        if current_q_step is None:
            return rate_param

        # Quantization is round(parameter_value / q_step) * q_step so we divide by q_step
        # to obtain the sent latent.
        current_sent_param = (parameter_value / current_q_step).view(-1)

        if ".weight" in parameter_name:
            sent_param.weight.append(current_sent_param)
        elif ".bias" in parameter_name:
            sent_param.bias.append(current_sent_param)
        else:
            print(f'Parameter name should include ".weight" or ".bias" Found: {parameter_name}')

    # For each sent parameters (e.g. all biases and all weights)
    # compute their cost with an exp-golomb coding.
    for field_wb in fields(DescriptorNN):
        weight_or_bias = field_wb.name
        param = getattr(sent_param, weight_or_bias)

        # If we do not have any parameter, there is no rate associated.
        # This can happens for the upsampling biases for instance
        if len(param) == 0:
            setattr(rate_param, weight_or_bias, 0.0)
            continue

        # Current exp-golomb count is None because the module is not yet
        # quantized. Return an all zero rate
        current_expgol_cnt = getattr(expgol_cnt, weight_or_bias)
        if current_expgol_cnt is None:
            return rate_param

        # Concatenate the list of parameters as a big one dimensional tensor
        param = torch.cat(param)

        # This will be pretty long! Could it be vectorized?
        # ! Todo: replace that with the actual encode_exp_golomb code?
        setattr(rate_param, weight_or_bias, exp_golomb_nbins(param, count=current_expgol_cnt))

    return rate_param


@torch.no_grad()
def exp_golomb_nbins(symbol: Tensor, count: int = 0) -> Tensor:
    """Compute the number of bits required to encode a Tensor of integers
    using an exponential-golomb code with exponent ``count``.
    This estimates the rate of an actual exp-golomb code with less than 0.5% mismatch.

    Args:
        symbol: Tensor to encode
        count (int, optional): Exponent of the exp-golomb code. Defaults to 0.

    Returns:
        Number of bits required to encode all the symbols.
    """
    # We encode the sign equiprobably at the end thus one more bit if symbol != 0
    nbins = 2 * torch.floor(torch.log2(2 * symbol.abs() / (2**count) + 1)) + count + (symbol != 0)
    res = nbins.sum()
    return res
