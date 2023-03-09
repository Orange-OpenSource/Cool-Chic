# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


"""Gather function used to quantize and entropy code an MLP."""

import itertools
import math
import time
import numpy as np
import torch

from collections import OrderedDict
from typing import Tuple
from torch import nn, Tensor
from torch.distributions import Laplace

from model_management.trainer import loss_fn
from utils.constants import MIN_SCALE_NN_WEIGHTS_BIAS, POSSIBLE_Q_STEP_NN


def quantize_param(x: Tensor, q_step: float) -> Tensor:
    """Quantize a tensor with a specified q_step

    Args:
        x (tensor): Tensor to be quantized (can be of any size)
        q_step (float): Quantization step

    Returns:
        tensor: quantized tensor
    """
    return torch.round(x / q_step) * q_step


def measure_gnd_truth_entropy(x: Tensor) -> float:
    """Compute the Shannon's entropy (i.e. optimal number of bit per symbol)
    for a tensor x.

    Args:
        x (tensor): tensor for which we want to estimate the entropy

    Returns:
        float: optimal number of bit per symbol of x
    """
    x = x.detach().cpu().numpy()
    # Get the number of occurrence for each different value
    _, counts = np.unique(x, return_counts=True)
    # Transform count to probability
    norm_counts = counts / counts.sum()
    # Entropy = sum( - p(x) * log2(p(x)) )
    entropy = -(norm_counts * np.log2(norm_counts)).sum()
    return entropy


def measure_laplace_rate(x: Tensor, q_step: float) -> float:
    """Get the rate associated to x when it is entropy-coded using a
    centered Laplace distribution.

    Args:
        x (tensor): tensor to be entropy-coded with a Laplace distribution.
            It can be of any dimension.

    Returns:
        float: costs in bit of the entire x tensor
    """
    # This are the integers values sent by the encoder after quantization
    x = quantize_param(x, q_step) / q_step
    # Scale = standard deviation / sqrt(2)
    distrib = Laplace(0., max(x.std().item() / math.sqrt(2), MIN_SCALE_NN_WEIGHTS_BIAS))
    # No value can cost more than 32 bits
    proba = torch.clamp(
        distrib.cdf(x + 0.5) - distrib.cdf(x - 0.5), min=2 ** -32, max=None
    )
    rate = -torch.log2(proba).sum()
    return rate


def quantize_model(fp_model: nn.Module, q_step_weight: float, q_step_bias: float) -> Tuple:
    """Quantize a full precision model fp_model at a given q_step. Return
    the quantized model and a 1D tensor gathering the quantized weights.
    !It only quantizes the MLP parameters!

    Args:
        fp_model (nn.Module): full precision model
        q_step_weight (float): quantization step for the weights
        q_step_bias (float): quantization step for the bias

    Returns:
        Tuple: quantized model and 1d tensor of the quantized weights and biases
    """
    q_model_param = {}
    q_weights, q_bias = [], []
    for k, v in fp_model.named_parameters():
        # ! Only parameters belonging to an object named "mlp" or "conv" are quantized
        if 'mlp' in k or 'conv' in k:
            # if k ends with '.w' it's a weight. If it's '.b' it's a bias
            if k.endswith('.w') or k.endswith('.weight'):
                q_v = quantize_param(v, q_step_weight)
                q_weights.append(q_v.view(-1).cpu())
            if k.endswith('.b') or k.endswith('.bias'):
                q_v = quantize_param(v, q_step_bias)
                q_bias.append(q_v.view(-1).cpu())

            q_model_param[k] = q_v
        else:
            q_model_param[k] = v

    q_weights = torch.cat(q_weights, dim=0)
    q_bias = torch.cat(q_bias, dim=0)
    # fp_model is now a quantized model
    fp_model.load_state_dict(q_model_param)
    return fp_model, q_weights, q_bias


@torch.no_grad()
def greedy_quantization(
    fp_model: nn.Module,
    img: Tensor,
    lmbda: float,
    verbose: bool = True,
) -> Tuple:
    """Find the best quantization steps for both MLP (ARM & Synthesis).
    This is perform in a brute force fashion by optimizing
        J(lambda) = Dist + lambda * (rate_latent + rate_model).

    Args:
        fp_model (nn.Module): Full precision model
        lmbda (float): Rate constraint.
        verbose (bool, optional): Print stuff. Default to True

    Returns:
        Tuple: Quantized model, best quantization step and rate network
    """
    start_time = time.time()

    # Prepare empty dictionaries to accomodate the results
    best_q_step, best_rate_mlp = {}, {}

    # Loop on the different module to quantize
    list_module_to_quantize = ['synthesis', 'arm']

    for current_module in list_module_to_quantize:
        # Prepare a copy of the model as it is before trying out the different quantization step
        fp_model_param = OrderedDict(
            (k, v.detach().clone())
            for k, v in fp_model.state_dict().items()
        )

        # Prepare a dummy best loss
        best_loss = 1e6

        # Just for printing stuff
        first_line_print = True

        # Try out different quantization step
        Q_STEP_LIST = POSSIBLE_Q_STEP_NN
        for q_step_weight, q_step_bias in itertools.product(Q_STEP_LIST, Q_STEP_LIST):
            # Reload the full precision model
            fp_model.load_state_dict(fp_model_param)

            # Get the current module to be quantized
            fp_module = getattr(fp_model, current_module)

            # Quantize the module and measure its rate
            q_module, q_module_w, q_module_b = quantize_model(
                fp_module, q_step_weight, q_step_bias
            )
            # Sum the rate of the bias and the weights
            rate_module = measure_laplace_rate(q_module_w, q_step_weight)
            rate_module += measure_laplace_rate(q_module_b, q_step_bias)

            # Replace current module weight with the quantized version
            # Set attribute fp_model.<current_module> = q_module
            setattr(fp_model, current_module, q_module)

            fp_model = fp_model.eval()
            model_out = fp_model()
            # Compute results
            loss, logs = loss_fn(model_out, img, lmbda, compute_logs=verbose, rate_mlp=rate_module)

            if verbose:
                first_row = 'q_step_w\tq_step_b'
                second_row = f'{q_step_weight:7.6f}\t{q_step_bias:7.6f}'
                for k, v in logs.items():
                    first_row += k + '\t'
                    second_row += f'{v:5.3f}\t'
                if first_line_print:
                    print(first_row)
                    first_line_print = False
                print(second_row)

            # Store best parameters
            if loss < best_loss:
                best_loss = loss
                best_model = OrderedDict((k, v.detach().clone()) for k, v in fp_model.state_dict().items())
                best_q_step[f'{current_module}_weight'] = q_step_weight
                best_q_step[f'{current_module}_bias'] = q_step_bias
                best_rate_mlp[current_module] = rate_module

        # Load best quantized parameters
        fp_model.load_state_dict(best_model)

    msg = f'Time Q_step greedy search: {time.time() - start_time:4.1f} seconds'
    msg += f' for {2 * (len(Q_STEP_LIST) ** 2)} combinations'
    print(f'\n{msg}\n')

    return fp_model, best_q_step, best_rate_mlp
