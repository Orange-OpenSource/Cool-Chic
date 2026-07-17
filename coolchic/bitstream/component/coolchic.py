import time
from typing import List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from coolchic.bitstream.component.armint import arm_to_fixed_point_param, fixed_point_arm
from coolchic.bitstream.component.latent import entropy_coding_latent_arm
from coolchic.bitstream.component.constants import (
    FIXED_POINT_DTYPE,
    N_FRAC_BIT_INTER_FT_CTX,
    WEIGHT_SHIFT,
)
from coolchic.bitstream.header.header import CoolChicHeader
from coolchic.bitstream.neuralnet.neuralnet import decode_network
from coolchic.bitstream.component.rangecoder import RangeCoder
from coolchic.component.core.coolchic import (
    instantiate_arm_from_cc_param,
    instantiate_common_randomness_from_cc_param,
    instantiate_ifce_from_cc_param,
    instantiate_syn_from_cc_param,
    instantiate_ups_from_cc_param,
)
from coolchic.component.core.upsampling import fixed_upsampling


def encode_decode_coolchic(
    header: CoolChicHeader,
    bytes_nn: bytes,
    mode: Literal["encode", "decode"],
    dec_bytes_latent: Optional[bytes] = None,
    enc_quantized_latent: Optional[List[Tensor]] = None,
    verbosity: int = 0,
) -> Union[Tensor, Optional[bytes]]:

    start_time = time.time()
    if mode == "encode":
        if enc_quantized_latent is None:
            raise ValueError(
                "Trying to encode cool_chic latent without indicating the quantized latent value. "
                "Found enc_quantized_latent=None. It should be a list of integer Tensor."
            )

    if mode == "decode":
        if dec_bytes_latent is None:
            raise ValueError(
                "Trying to encode cool_chic latent with dec_bytes_latent=None. "
                "The argument dec_bytes_latent should represent the bytes of the bitstream."
            )

    param = header.get_coolchic_parameter()

    # Reconstruct the network architecture
    empty_arm = instantiate_arm_from_cc_param(param)
    empty_syn = instantiate_syn_from_cc_param(param)
    empty_ups = instantiate_ups_from_cc_param(param)
    empty_ifce = instantiate_ifce_from_cc_param(param)

    # Decode the network
    descriptors_nn = {
        k: header.get_value(k) for k in ["nn_expgol_cnt", "nn_n_bytes", "nn_q_step", "nn_n_bit_pad"]
    }

    decoded_module, _ = decode_network(
        bytes_nn, descriptors_nn, empty_arm, empty_ups, empty_syn, empty_ifce
    )
    time_neural_net = time.time() - start_time
    start_time = time.time()

    arm_w_fp, arm_b_fp, arm_w_stab_fp, arm_b_stab_fp = arm_to_fixed_point_param(
        decoded_module.get("arm"),
        header.get_value("nn_q_step").arm,
        n_inter_ft_ctx=param.output_feature_ifce,
        subtract_last_layer=True,
    )

    # Encode the different 2d latent grids one after the other
    # Loop on the different resolutions
    range_coder = RangeCoder()
    coded_latent = []

    if mode == "decode":
        range_coder.load_bitstream(dec_bytes_latent)

    # Start from the lowest resolution
    time_ifce = 0
    for idx_latent in range(param.n_latent_grids - 1, -1, -1):
        h_i, w_i = param.size_per_latent[idx_latent][-2:]

        start_time_ifce = time.time()

        # Smallest i.e. first latent to be decoded
        if idx_latent == (param.n_latent_grids - 1):
            ups_coded_latent = torch.zeros((1, 1, h_i, w_i), dtype=FIXED_POINT_DTYPE)
        else:
            # Upscale the already (de)coded latent to use as context by the ifce
            ups_coded_latent, _ = fixed_upsampling(coded_latent, mode="nearest")
            ups_coded_latent = ups_coded_latent.to(FIXED_POINT_DTYPE)

        # Flatten for the upcoming IFCE forward
        flat_ups_coded_latent = rearrange(ups_coded_latent, "1 c h w -> (h w) c")

        if param.flag_ifce:
            # No IFCE for this latent
            if param.input_features_ifce[idx_latent] == 0:
                h_ups_coded_latent, w_ups_coded_latent = ups_coded_latent.size()[-2:]
                context_inter_feature = torch.zeros(
                    (h_ups_coded_latent * w_ups_coded_latent, param.output_feature_ifce),
                    dtype=FIXED_POINT_DTYPE,
                )
            else:
                w_ifce, b_ifce, w_stab_ifce, b_stab_ifce = arm_to_fixed_point_param(
                    decoded_module.get("ifce").arms[
                        decoded_module.get("ifce").index_to_arm[idx_latent]
                    ],
                    header.get_value("nn_q_step").ifce,
                    # We don't subtract -4 to the last layer of the inter feature arm
                    subtract_last_layer=False,
                    n_inter_ft_ctx=0,
                    no_residual_layer=True,
                )

                context_inter_feature = fixed_point_arm(
                    flat_ups_coded_latent,
                    w_ifce,
                    b_ifce,
                    w_stab_ifce,
                    b_stab_ifce,
                    output_shift=2 * WEIGHT_SHIFT - N_FRAC_BIT_INTER_FT_CTX,
                )

            context_inter_feature = rearrange(
                context_inter_feature,
                "(h w) c -> 1 c h w",
                h=ups_coded_latent.size()[2],
                w=ups_coded_latent.size()[3],
            )

            # Interpolate one last time to reach the resolution of spatial_latent i
            context_inter_feature = F.interpolate(
                context_inter_feature.to(torch.float), scale_factor=2, mode="nearest"
            ).to(FIXED_POINT_DTYPE)
            # Trim additional pixel due to the upsampling
            context_inter_feature = context_inter_feature[:, :, :h_i, :w_i]
        else:
            context_inter_feature = None

        time_ifce += time.time() - start_time_ifce

        latent_i = entropy_coding_latent_arm(
            enc_quantized_latent[idx_latent].to(FIXED_POINT_DTYPE) if mode == "encode" else None,
            context_inter_feature,
            (h_i, w_i),
            arm_w_fp,
            arm_b_fp,
            arm_w_stab_fp,
            arm_b_stab_fp,
            range_coder,
            mode=mode,
            n_spatial_context=param.spatial_context_arm,
        )

        # Insert at the beginning so that coded_latent[0] is the biggest latent
        coded_latent.insert(0, latent_i.to(torch.float32))

    if mode == "encode":
        enc_bytes_latent = range_coder.get_bitstream_bytes()
        header.set_value("n_bytes_latent", len(enc_bytes_latent))
    time_latent = time.time() - start_time
    start_time = time.time()

    # Filter hyperlatent out from the synthesis
    coded_latent = [x for x, m in zip(coded_latent, param.flag_is_hyperlatent) if not m]

    synthesis_input = decoded_module.get("upsampling")(coded_latent)

    if header.get_value("flag_common_randomness"):
        cr_latent = instantiate_common_randomness_from_cc_param(param)
        ups_noise, _ = fixed_upsampling(cr_latent)
        ups_noise = F.interpolate(ups_noise, size=param.img_size, mode="bicubic")
        synthesis_input = torch.cat([synthesis_input, ups_noise], dim=1)

    synthesis_output = decoded_module.get("synthesis")(synthesis_input)
    # Upsample the output of the synthesis with a bicubic if required
    synthesis_output = F.interpolate(
        synthesis_output, size=param.img_size, mode=param.final_upsampling_type
    )

    # Trim out additional pixels due to the final upsampling
    synthesis_output = synthesis_output[:, :, : param.img_size[0], : param.img_size[1]]

    if mode == "encode":
        coolchic_bytes = header.to_bytes() + bytes_nn + enc_bytes_latent
    else:
        coolchic_bytes = None

    time_syn = time.time() - start_time
    start_time = time.time()

    if verbosity:
        print(header.pretty_string())
    if verbosity >= 2:
        print(f"{time_neural_net:6.2f} {time_ifce:6.2f} {time_latent:6.2f} {time_syn:6.2f} ")

    return synthesis_output, coolchic_bytes
