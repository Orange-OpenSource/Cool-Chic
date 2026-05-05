# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import time
from typing import Tuple

import torch
from torch.nn.utils import clip_grad_norm_

from coolchic.component.frame import FrameEncoder
from coolchic.io.format.yuv import convert_420_to_444
from coolchic.training.loss import loss_function
from coolchic.training.presets import TrainerPhase
from coolchic.training.soap import SOAP
from coolchic.training.test import test
from coolchic.utils.codingstructure import Frame


# Custom scheduling function for the soft rounding temperature and the noise parameter
def _linear_schedule(
    initial_final_value: Tuple[float, float], cur_itr: float, max_itr: float
) -> float:
    """Linearly schedule a function to go from initial_value at cur_itr = 0 to
    final_value when cur_itr = max_itr.

    Args:
        initial_final_value (Tuple[float, float]): Initial and final values for the scheduling
        cur_itr (float): Current iteration index
        max_itr (float): Total number of iterations

    Returns:
        float: The linearly scheduled value @ iteration number cur_itr
    """
    assert cur_itr >= 0 and cur_itr <= max_itr, (
        f"Linear scheduling from 0 to {max_itr} iterations"
        " except to have a current iterations between those two values."
        f" Found cur_itr = {cur_itr}."
    )

    initial_value, final_value = initial_final_value
    return cur_itr * (final_value - initial_value) / max_itr + initial_value


def train(
    frame_encoder: FrameEncoder,
    frame: Frame,
    training_phase: TrainerPhase,
) -> FrameEncoder:
    """Train a ``FrameEncoder`` and return the updated module. This function is
    supposed to be called any time we want to optimize the parameters of a
    FrameEncoder, either during the warm-up (competition of multiple possible
    initializations) or during of the stages of the actual training phase.

    The module is optimized according to the following loss function:

    .. math::

        \\mathcal{L} = \\mathrm{D}(\hat{\\mathbf{x}}, \\mathbf{x}) + \\lambda
        \\mathrm{R}(\hat{\\mathbf{x}}), \\text{ with } \\begin{cases}
            \\mathbf{x} & \\text{the original image}\\\\ \\hat{\\mathbf{x}} &
            \\text{the coded image}\\\\ \\mathrm{R}(\\hat{\\mathbf{x}}) &
            \\text{A measure of the rate of } \\hat{\\mathbf{x}}\\\\
            \\mathrm{D}(\hat{\\mathbf{x}}, \\mathbf{x})  & \\text{A distortion
            metric specified by \\texttt{--tune} and \\texttt{--alpha}}
        \\end{cases}

    Args:
        frame_encoder: Module to be trained.
        frame: The original image to be compressed and its references.
        training_phase: Hyperparameters of the current training phase.
    Returns:
        The trained frame encoder.
    """
    start_time = time.time()

    # We train with dense reference!
    for idx_ref, ref_i in enumerate(frame.refs_data):
        if ref_i.frame_data_type == "yuv420":
            frame.refs_data[idx_ref].data = convert_420_to_444(ref_i.data)
            frame.refs_data[idx_ref].frame_data_type = "yuv444"

    raw_references_444 = [ref_i.data for ref_i in frame.refs_data]

    # ------ Keep track of the best loss and model
    # Perform a first test to get the current best logs (it includes the loss)
    initial_encoder_logs = test(
        frame_encoder=frame_encoder,
        frame=frame,
        dist_weight=training_phase.dist_weight,
        lmbda=training_phase.lmbda,
    )
    encoder_logs_best = initial_encoder_logs
    best_model = frame_encoder.get_param()

    frame_encoder.set_to_train()

    # ------ Build the list of parameters to optimize
    all_parameters = []
    weight_parameters = []
    latent_parameters = []
    for k, v in frame_encoder.named_parameters():
        # Always ignore the output transform parameter
        if "synthesis.output_transform" in k:
            continue

        all_parameters.append(v)
        if "latent" in k:
            latent_parameters.append(v)
        else:
            weight_parameters.append(v)

    optimizer = SOAP(
        [
            # Network parameters: full SOAP
            {
                "params": weight_parameters,
                "lr": training_phase.lr,
                "betas": (0.95, 0.95),
                "precondition_frequency": 10,
                "max_precond_dim": 256,
                "merge_dims": False,
                "precondition_1d": False,
                "weight_decay": 0.01,
            },
            # Latent grids: Adam-like
            {
                "params": latent_parameters,
                "lr": training_phase.lr,
                "betas": (0.9, 0.999),
                "precondition_frequency": 1,  # irrelevant here
                "max_precond_dim": 0,  # DISABLE SOAP
                "merge_dims": False,
                "precondition_1d": False,
                "weight_decay": 0.0,
            },
        ]
    )

    # best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    if training_phase.schedule_lr:
        # TODO: I'd like to use an explicit function for this scheduler
        learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_phase.max_itr / training_phase.freq_valid,
            eta_min=0.00001,
            last_epoch=-1,
        )
    else:
        learning_rate_scheduler = None

    # Initialize soft rounding temperature and noise parameter
    cur_softround_temperature = _linear_schedule(
        training_phase.softround_temperature,
        0,
        training_phase.max_itr,
    )
    device = (
        frame.data.data.device
        if frame.data.frame_data_type != "yuv420"
        else frame.data.data.get("y").device
    )
    cur_softround_temperature = torch.tensor(cur_softround_temperature, device=device)

    cur_noise_parameter = _linear_schedule(
        training_phase.noise_parameter, 0, training_phase.max_itr
    )
    cur_noise_parameter = torch.tensor(cur_noise_parameter, device=device)

    cnt_record = 0
    show_col_name = True  # Only for a pretty display of the logs
    # Slightly faster to create the list once outside of the loop
    all_parameters = [x for x in frame_encoder.parameters()]

    for cnt in range(training_phase.max_itr):
        # print(sum(v.abs().sum() for _, v in best_model.items()))

        # ------- Patience mechanism
        if cnt - cnt_record > training_phase.patience:
            if training_phase.schedule_lr:
                # reload the best model so far
                frame_encoder.set_param(best_model)
                # optimizer.load_state_dict(best_optimizer_state)

                current_lr = learning_rate_scheduler.state_dict()["_last_lr"][0]
                # actualise the best optimizer lr with current lr
                for g in optimizer.param_groups:
                    g["lr"] = current_lr

                cnt_record = cnt
            else:
                # exceeding the training_phase.patience level ends the phase
                break

        # ------- Actual optimization
        # This is slightly faster than optimizer.zero_grad()
        for param in all_parameters:
            param.grad = None

        # forward / backward
        out_forward = frame_encoder.forward(
            reference_frames=raw_references_444,
            quantizer_noise_type=training_phase.quantizer_noise_type,
            quantizer_type=training_phase.quantizer_type,
            soft_round_temperature=cur_softround_temperature,
            noise_parameter=cur_noise_parameter,
        )

        decoded_image = out_forward.decoded_image
        target_image = frame.data.data

        loss_function_output = loss_function(
            decoded_image=decoded_image,
            rate_latent_bit=out_forward.rate,
            target_image=target_image,
            dist_weight=training_phase.dist_weight,
            lmbda=training_phase.lmbda,
            total_rate_nn_bit=0.0,
            compute_logs=False,
        )
        loss_function_output.loss.backward()

        # the weight parameters gradients are clipped, it is not necessary for the latents
        clip_grad_norm_(weight_parameters, 0.1, norm_type=2.0, error_if_nonfinite=False)

        optimizer.step()

        frame_encoder.encoder_monitor.increment_iterations(1)

        # ------- Validation
        # Each freq_valid iteration or at the end of the phase, compute validation loss and log stuff
        encoder_logs = None
        if ((cnt + 1) % training_phase.freq_valid == 0) or (cnt + 1 == training_phase.max_itr):
            #  a. Update iterations counter and training time and test model
            frame_encoder.encoder_monitor.increment_time(time.time() - start_time)
            start_time = time.time()

            # b. Test the model and check whether we've beaten our record
            encoder_logs = test(
                frame_encoder=frame_encoder,
                frame=frame,
                dist_weight=training_phase.dist_weight,
                lmbda=training_phase.lmbda,
            )

            if encoder_logs.loss < encoder_logs_best.loss:
                # Save best model
                best_model = frame_encoder.get_param()
                # best_optimizer_state = copy.deepcopy(optimizer.state_dict())

                # ========================= reporting ========================= #
                this_phase_dist_gain = encoder_logs.dist_db - initial_encoder_logs.dist_db
                this_phase_bpp_gain = (
                    encoder_logs.total_rate_latent_bpp - initial_encoder_logs.total_rate_latent_bpp
                )

                log_new_record = ""
                log_new_record += f"{this_phase_bpp_gain:+6.3f} bpp "
                log_new_record += f"{this_phase_dist_gain:+6.3f} db"
                # ========================= reporting ========================= #

                # Update new record
                encoder_logs_best = encoder_logs
                cnt_record = cnt
            else:
                log_new_record = ""

            # Show column name a single time
            additional_data = {
                "lr": f"{training_phase.lr if not training_phase.schedule_lr else learning_rate_scheduler.get_last_lr()[0]:.4f}",
                # "optim": ",".join(training_phase.optimized_module),
                "patience": (training_phase.patience - cnt + cnt_record)
                // training_phase.freq_valid,
                "q_type": f"{training_phase.quantizer_type:10s}",
                "sr_temp": f"{cur_softround_temperature:.3f}",
                "n_type": f"{training_phase.quantizer_noise_type[:8]:10s}",
                "noise": f"{cur_noise_parameter:.2f}",
                "record": log_new_record,
            }

            print(
                encoder_logs.pretty_string(
                    show_col_name=show_col_name,
                    mode="short",
                    additional_data=additional_data,
                )
            )
            show_col_name = False

            # Update soft rounding temperature and training_phase.noise_parameter
            cur_softround_temperature = _linear_schedule(
                training_phase.softround_temperature,
                cnt,
                training_phase.max_itr,
            )
            cur_softround_temperature = torch.tensor(cur_softround_temperature, device=device)

            cur_noise_parameter = _linear_schedule(
                training_phase.noise_parameter,
                cnt,
                training_phase.max_itr,
            )
            cur_noise_parameter = torch.tensor(cur_noise_parameter, device=device)

            if training_phase.schedule_lr:
                learning_rate_scheduler.step()

            frame_encoder.set_to_train()

    # At the end of the training, we load the best model
    frame_encoder.set_param(best_model)
    return frame_encoder
