# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import time
import torch
from typing import Tuple, Dict
from collections import OrderedDict
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.cool_chic import EncoderOutput
from utils.logging import format_results_loss

def save_model(model: nn.Module, path: str):
    """Save an entire model @ path

    Args:
        model (nn.Module): The model to save
        path (str): Where to save
    """
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module,path: str) -> nn.Module:
    """Load a model from path

    Args:
        model (nn.Module): The model to fill with the pre-trained weights
        path (str): Path of the file where the model is saved

    Returns:
        nn.Module: The loaded module
    """
    model.load_state_dict(torch.load(path))
    return model


@torch.jit.script
def mse_fn(x: Tensor, y: Tensor) -> Tensor:
    """Torch scripted version of the MSE. Compute the Mean Squared Error
    of two tensors with arbitrary dimension.

    Args:
        x and y (Tensor): Compute the MSE of x and y.

    Returns:
        Tensor: One element tensor containing the MSE of x and y.
    """
    return torch.pow((x - y), 2.0).mean()


def loss_fn(
    out_forward: EncoderOutput,
    target: Tensor,
    lmbda: float,
    compute_logs: bool = False,
    rate_mlp: float = 0.,
) -> Tuple:
    """Compute the loss and other quantities from the network output out_forward

    Args:
        out_forward (EncoderOutput): Encoder-side output data.
        target (tensor): [1, 3, H, W] tensor of the ground truth image
        lmbda (float): Rate constraint
        compute_logs (bool, Optional): If true compute additional quantities. This
            includes the MS-SSIM Which in turn requires that out_forward describes the
            entire image. Default to False.
        rate_mlp (float, Optional): Rate of the network if it needs to be present in the
            loss computation. Expressed in bit! Default to 0.

    Returns:
        Tuple: return loss and log dictionary (only if compute_logs)
    """
    x_hat = out_forward.get('x_hat')
    n_pixels = x_hat.size()[-2] * x_hat.size()[-1]
    mse = mse_fn(x_hat, target)
    rate_bpp = (out_forward.get('rate_y') + rate_mlp) / n_pixels
    loss = mse + lmbda * rate_bpp

    if compute_logs:
        logs = {
            'loss': loss.detach() * 1000,
            'psnr': 10. * torch.log10(1. / mse.detach()),
            'rate_mlp': rate_mlp / n_pixels,        # Rate MLP in bpp
        }

        # Append the different rates (in bpp) to the log
        for k, v in out_forward.items():
            if 'rate' not in k:
                continue
            # Ignore lists which are due to the comprehensive rate_per_grid tensor
            if isinstance(v, list):
                continue
            if isinstance(v, Tensor):
                v = v.detach().item()
            logs[k] = v / n_pixels

        logs['rate_all_bpp'] = logs.get('rate_mlp') + logs.get('rate_y')

    else:
        logs = None

    return loss, logs


def train(
    model: nn.Module,
    target: Tensor,
    n_itr: int = int(5e3),
    start_lr: float = 1e-2,
    lmbda: float = 5e-3,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train an INR codec

    Args:
        model (nn.Module): INRCodec module already instantiated
        target (tensor): [1, C, H, W] tensor representing the image to encode
        n_itr (int, optional): Number of iterations. Defaults to int(5e3).
        start_lr (float, optional): Initial learning rate. Defaults to 1e-2.
        lmbda (float, optional): Rate constraint. Loss is D + lmbda * R. Defaults to 5e-3.

    Returns:
        nn.Module: the trained model
    """

    model = model.train()

    # Logs some useful training statistics inside this dictionary
    training_stat_logs = {}

    # Create optimizer (it takes a parameter list as inputs)
    PATIENCE = 15
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=PATIENCE, verbose=True)

    # Keep track of the best model found during training
    best_loss = 1e6
    best_model = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())

    # Use the uniform noise proxy for the quantization at the beginning of the training
    use_ste_quant = False

    # Only for printing
    first_line_print = True

    start_time = time.time()
    overall_start_time = start_time
    for cnt in range(n_itr):
        # This is slightly faster than optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        out_forward = model(use_ste_quant = use_ste_quant)
        loss, _ = loss_fn(out_forward, target, lmbda, compute_logs=False)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10., norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        # Each 100 iteration, compute validation loss and log stuff
        if (cnt + 1) % 100 == 0 or cnt == (n_itr - 1):
            model = model.eval()
            # Valid on the whole picture to feed the patience-based scheduler
            with torch.no_grad():
                out_forward = model()
                loss, logs = loss_fn(out_forward, target, lmbda, compute_logs=True)

                n_bad_epoch_before = scheduler.num_bad_epochs
                scheduler.step(loss)
                n_bad_epoch_after = scheduler.num_bad_epochs

            if loss < best_loss:
                for k, v in model.state_dict().items():
                    best_model[k].copy_(v)
                best_loss = loss.detach()
                log_new_record = '>>> New record!'
            else:
                log_new_record = ''

            # If the counter of bad epochs is was equal to PATIENCE and then to 0
            # it means that we're in two situations:
            #       - We've reached the patience threshold and decreased the learning rate;
            #       - We've just beaten our record at the last moment (and stored the best_model).
            # In both case, we can reset our model to the last best one
            if (n_bad_epoch_before == PATIENCE) and (n_bad_epoch_after == 0):
                print('Resetting model to last best model')
                model.load_state_dict(best_model)

            # ====================== Print some logs ====================== #
            logs['iteration'] = cnt + 1
            logs['time_sec'] = time.time() - start_time

            print(format_results_loss(logs, col_name=first_line_print) + log_new_record)
            first_line_print = False
            start_time = time.time()
            # ====================== Print some logs ====================== #

            # Restore training mode
            model = model.train()

        if not(use_ste_quant) and optimizer.param_groups[0]['lr'] < 9e-4:
            print('Switching from uniform noise to STE for quantization')
            use_ste_quant = True

        # Stop training if the learning rate becomes too small
        if optimizer.param_groups[0]['lr'] < 4e-4:
            break

    # Load best model
    model.load_state_dict(best_model)

    training_stat_logs = {
        'training_time_second': time.time() - overall_start_time,
        'n_iteration': cnt,
        'loss': best_loss,
    }

    return model, training_stat_logs
