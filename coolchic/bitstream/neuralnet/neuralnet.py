# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

from dataclasses import fields
from typing import Dict, Optional, Tuple, Union

import torch

from coolchic.bitstream.component.constants import FIXED_POINT_DTYPE
from coolchic.bitstream.neuralnet.expgolomb import decode_exp_golomb, encode_exp_golomb
from coolchic.component.core.coolchic import CoolChicEncoder
from coolchic.component.core.arm import Arm, Ifce
from coolchic.component.core.synthesis import Synthesis
from coolchic.component.core.types import DescriptorCoolChic, DescriptorNN
from coolchic.component.core.upsampling import Upsampling
from coolchic.nnquant.expgolomb import POSSIBLE_EXP_GOL_COUNT
from coolchic.nnquant.quantstep import POSSIBLE_Q_STEP


def encode_network(cc_enc: CoolChicEncoder) -> Tuple[bytes, Dict[str, DescriptorCoolChic]]:
    """Encode all the neural network of a CoolChicEncoder into bytes. Also return a dictionary
    containing the information about the binarisation (number of alignement bits, q steps,
    exp-golomb parameters and number of bits for the neural networks).

    Args:
        cc_enc (CoolChicEncoder): Cool-chic module containing the neural network to write to bytes.

    Returns:
        Tuple[bytes, Dict[str, DescriptorCoolChic]]: Bytes corresponding to the neural network and
            description parameters.
    """

    nn_expgol_cnt = DescriptorCoolChic()
    nn_q_step = DescriptorCoolChic()
    nn_n_bit_pad = DescriptorCoolChic()

    data_bytes = b""

    all_q_param = []
    all_count = []

    for field_nn in fields(DescriptorCoolChic):
        # Either arm, upsampling or synthesis
        module_name = field_nn.name
        module_to_encode = getattr(cc_enc, module_name)

        for field_wb in fields(DescriptorNN):
            # Either weight or bias
            weight_or_bias = field_wb.name

            if module_to_encode is None:
                # Select any quantization step and expgol cnt parameter
                q_step = POSSIBLE_Q_STEP.get_value(module_name, weight_or_bias)[0]
                expgol_cnt = POSSIBLE_EXP_GOL_COUNT.get_value(module_name, weight_or_bias)[0]
            else:
                # Retrieve the quantization step and its index
                q_step = cc_enc.nn_q_step.get_value(module_name, weight_or_bias)
                expgol_cnt = cc_enc.nn_expgol_cnt.get_value(module_name, weight_or_bias)

                # Retrieve the param and quantize them.
                param_dict = module_to_encode.get_param(which=weight_or_bias)
                param = torch.cat([param_val.flatten() for _, param_val in param_dict.items()])
                q_param = torch.round(param / q_step).to(torch.int32)

                all_q_param += q_param.tolist()
                all_count += [expgol_cnt for _ in range(q_param.numel())]

            # Store values relative to the entropy coding of the neural network parameters.
            # This will be written into the header
            nn_q_step.set_value(q_step, module_name, weight_or_bias)
            nn_expgol_cnt.set_value(expgol_cnt, module_name, weight_or_bias)

    data_bytes, nn_n_bit_pad = encode_exp_golomb(all_q_param, all_count)

    # This dict contains all the information we'll need to put in the header besides
    # the raw entropy coded bytes
    all_descriptors = {
        "nn_expgol_cnt": nn_expgol_cnt,
        "nn_q_step": nn_q_step,
        "nn_n_bytes": len(data_bytes),
        "nn_n_bit_pad": nn_n_bit_pad,
    }
    return data_bytes, all_descriptors


def decode_network(
    data_bytes: bytes,
    descriptors: Dict[str, DescriptorCoolChic],
    empty_arm: Arm,
    empty_ups: Upsampling,
    empty_syn: Synthesis,
    empty_ifce: Optional[Ifce] = None,
) -> Union[Dict[str, Union[Arm, Upsampling, Synthesis, Ifce]], bytes]:
    """Decode the bytes corresponding to Neural Networks weights and biases. Put them inside
    empty modules and return the modules filled with parameters.

    Args:
        data_bytes (bytes): Bytes encoding the neural network parameters.
        descriptors (Dict[str, DescriptorCoolChic]): information about the binarisation
            (number of alignement bits, q steps, exp-golomb parameters and number of bits for
            the neural networks)
        empty_arm (Arm): ARM with the proper architecture, ready to be filled with params.
        empty_ups (Upsampling): Upsampling with the proper architecture, ready to be filled with params.
        empty_syn (Synthesis): Synthesis with the proper architecture, ready to be filled with params.
        empty_ifce (Optional[Ifce], optional): Synthesis with the proper architecture, ready to be filled
            with params. None if no IFCE.

    Returns:
        Union[Dict[str, Union[Arm, Upsampling, Synthesis, Ifce]], bytes]: A dictionary containing
            the decoded neural networks and the remaining bytes of the bitstream.
    """
    # Step 1: compute the total number of NN parameters to read and create
    # the count Tensor indicating the exp-golomb order of each parameter
    count = []
    for field_nn in fields(DescriptorCoolChic):
        # Either arm, upsampling or synthesis
        module_name = field_nn.name
        match module_name:
            case "arm":
                module_to_fill = empty_arm
            case "synthesis":
                module_to_fill = empty_syn
            case "upsampling":
                module_to_fill = empty_ups
            case "ifce":
                module_to_fill = empty_ifce

        if module_to_fill is None:
            continue

        for field_wb in fields(DescriptorNN):
            # Either weight or bias
            weight_or_bias = field_wb.name

            total_param_to_read = sum(
                [v.numel() for _, v in module_to_fill.get_param(which=weight_or_bias).items()]
            )

            count += [
                int(descriptors.get("nn_expgol_cnt").get_value(module_name, weight_or_bias))
                for _ in range(total_param_to_read)
            ]

    # Step 2: Decode all the parameters
    all_param = decode_exp_golomb(data_bytes, descriptors.get("nn_n_bit_pad"), count)

    decoded_modules = {}

    for field_nn in fields(DescriptorCoolChic):
        # Either arm, upsampling or synthesis
        module_name = field_nn.name
        match module_name:
            case "arm":
                module_to_fill = empty_arm
            case "synthesis":
                module_to_fill = empty_syn
            case "upsampling":
                module_to_fill = empty_ups
            case "ifce":
                module_to_fill = empty_ifce

        if module_to_fill is None:
            continue

        # Ignore the requires_grad that might be present
        module_to_fill.requires_grad_(False)
        for field_wb in fields(DescriptorNN):
            # Either weight or bias
            weight_or_bias = field_wb.name

            # Store the decoded parameters into the module to fill
            loaded_param = {}
            for k, v in module_to_fill.get_param(which=weight_or_bias).items():
                n_param_to_read = v.numel()
                cur_param = all_param[:n_param_to_read]
                all_param = all_param[n_param_to_read:]

                cur_param = torch.tensor(cur_param, dtype=FIXED_POINT_DTYPE)
                # We don't do the inverse quantization for the arm as it uses integer arithmetic
                if module_name not in ["arm", "ifce"]:
                    q_step = descriptors.get("nn_q_step").get_value(module_name, weight_or_bias)
                    cur_param = cur_param.to(torch.float32) * q_step

                # # Work from PyTorch 2.3 onwards
                # loaded_param[k] = cur_param.clone().reshape_as(v)

                # Work around for Pytorch <= 2.2: load_state_dict does not preserve requires_grad value
                # the requires_grad so we have to explicitely set requires_grad to False
                loaded_param[k] = torch.nn.Parameter(
                    cur_param.clone().reshape_as(v), requires_grad=False
                )

            # Can not be strict as we're loading separately the weights and biases
            # assign=True to maintain the type of the data i.e. int32 for ARM
            module_to_fill.load_state_dict(loaded_param, strict=False, assign=True)

        decoded_modules[module_name] = module_to_fill
    return decoded_modules, data_bytes
