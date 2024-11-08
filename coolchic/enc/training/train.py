# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import copy
import time
from typing import List, Tuple

import torch
from torch.nn.utils import clip_grad_norm_

from enc.utils.manager import FrameEncoderManager
from enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
)
from enc.component.frame import FrameEncoder
from enc.training.loss import loss_function
from enc.training.test import test
from enc.utils.codingstructure import Frame
from enc.training.presets import MODULE_TO_OPTIMIZE


# Custom scheduling function for the soft rounding temperature and the noise parameter
def _linear_schedule(
    initial_value: float, final_value: float, cur_itr: float, max_itr: float
) -> float:
    """Linearly schedule a function to go from initial_value at cur_itr = 0 to
    final_value when cur_itr = max_itr.

    Args:
        initial_value (float): Initial value for the scheduling
        final_value (float): Final value for the scheduling
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

    return cur_itr * (final_value - initial_value) / max_itr + initial_value


def train(
    frame_encoder: FrameEncoder,
    frame: Frame,
    frame_encoder_manager: FrameEncoderManager,
    start_lr: float = 1e-2,
    cosine_scheduling_lr: bool = True,
    max_iterations: int = 10000,
    frequency_validation: int = 100,
    patience: int = 10,
    optimized_module: List[MODULE_TO_OPTIMIZE] = ["all"],
    quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
    quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
    softround_temperature: Tuple[float, float] = (0.3, 0.2),
    noise_parameter: Tuple[float, float] = (2.0, 1.0),
) -> FrameEncoder:
    """Train a ``FrameEncoder`` and return the updated module. This function is
    supposed to be called any time we want to optimize the parameters of a
    FrameEncoder, either during the warm-up (competition of multiple possible
    initializations) or during of the stages of the actual training phase.

    The module is optimized according to the following loss function:

    .. math::

        \\mathcal{L} = ||\\mathbf{x} - \hat{\\mathbf{x}}||^2 + \\lambda
        \\mathrm{R}(\hat{\\mathbf{x}}), \\text{ with } \\begin{cases}
            \\mathbf{x} & \\text{the original image}\\\\ \\hat{\\mathbf{x}} &
            \\text{the coded image}\\\\ \\mathrm{R}(\\hat{\\mathbf{x}}) &
            \\text{A measure of the rate of } \\hat{\\mathbf{x}}
        \\end{cases}

    .. warning::

        The parameter ``frame_encoder_manager`` tracking the encoding time of
        the frame (``total_training_time_sec``) and the number of encoding
        iterations (``iterations_counter``) is modified **in place** by this
        function.

    Args:
        frame_encoder: Module to be trained.
        frame: The original image to be compressed and its references.
        frame_encoder_manager: Contains (among other things) the rate
            constraint :math:`\\lambda`. It is also used to track the total
            encoding time and encoding iterations. Modified in place.
        start_lr: Initial learning rate. Either constant for the entire
            training or schedule using a cosine scheduling, see below for more
            details. Defaults to 1e-2.
        cosine_scheduling_lr: True to schedule the learning
            rate from ``start_lr`` at iteration n°0 to 0 at iteration
            n° ``max_iterations``. Defaults to True.
        max_iterations: Do at most ``max_iterations`` iterations.
            The actual number of iterations can be made smaller through the
            patience mechanism. Defaults to 10000.
        frequency_validation: Check (and print) the performance
            each ``frequency_validation`` iterations. This drives the patience
            mechanism. Defaults to 100.
        patience: After ``patience`` iterations without any
            improvement to the results, exit the training. Patience is disabled
            by setting ``patience = max_iterations``. If patience is used alongside
            cosine_scheduling_lr, then it does not end the training. Instead,
            we simply reload the best model so far once we reach the patience,
            and the training continue. Defaults to 10.
        optimized_module: List of modules to be optimized. Most often you'd
            want to use ``optimized_module = ['all']``. Defaults to ``['all']``.
        quantizer_type: What quantizer to
            use during training. See :doc:`encoder/component/core/quantizer.py
            <../component/core/quantizer>` for more information. Defaults to
            ``"softround"``.
        quantizer_noise_type: The random noise used by the quantizer. More
            information available in
            :doc:`encoder/component/core/quantizer.py
            <../component/core/quantizer>`. Defaults to ``"kumaraswamy"``.
        softround_temperature: The softround temperature is linearly scheduled
            during the training. At iteration n° 0 it is equal to
            ``softround_temperature[0]`` while at iteration n° ``max_itr`` it is
            equal to ``softround_temperature[1]``. Note that the patience might
            interrupt the training before it reaches this last value.
            Defaults to (0.3, 0.2).
        noise_parameter: The random noise temperature is linearly scheduled
            during the training. At iteration n° 0 it is equal to
            ``noise_parameter[0]`` while at iteration n° ``max_itr`` it is equal
            to ``noise_parameter[1]``. Note that the patience might interrupt
            the training before it reaches this last value. Defaults to (2.0,
            1.0).

    Returns:
        The trained frame encoder.
    """

    start_time = time.time()

    frame.upsample_reference_to_444()
    raw_references = [ref_i.data for ref_i in frame.refs_data]

    # ------ Keep track of the best loss and model
    # Perform a first test to get the current best logs (it includes the loss)
    initial_encoder_logs = test(frame_encoder, frame, frame_encoder_manager)
    encoder_logs_best = initial_encoder_logs
    best_model = frame_encoder.get_param()

    frame_encoder.set_to_train()

    # ------ Build the list of parameters to optimize
    # Iteratively construct the list of required parameters... This is kind of a
    # strange syntax, which has been found quite empirically

    parameters_to_optimize = []

    if "arm" in optimized_module:
        parameters_to_optimize += [*frame_encoder.coolchic_encoder.arm.parameters()]
    if "upsampling" in optimized_module:
        parameters_to_optimize += [
            *frame_encoder.coolchic_encoder.upsampling.parameters()
        ]
    if "synthesis" in optimized_module:
        parameters_to_optimize += [
            *frame_encoder.coolchic_encoder.synthesis.parameters()
        ]
    if "latent" in optimized_module:
        parameters_to_optimize += [
            *frame_encoder.coolchic_encoder.latent_grids.parameters()
        ]
    if "all" in optimized_module:
        parameters_to_optimize = frame_encoder.parameters()

    optimizer = torch.optim.Adam(parameters_to_optimize, lr=start_lr)
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    if cosine_scheduling_lr:
        # TODO: I'd like to use an explicit function for this scheduler
        learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_iterations / frequency_validation,
            eta_min=0.00001,
            last_epoch=-1,
        )
    else:
        learning_rate_scheduler = None

    # Initialize soft rounding temperature and noise parameter
    cur_softround_temperature = _linear_schedule(
        softround_temperature[0],
        softround_temperature[1],
        0,
        max_iterations,
    )
    cur_noise_parameter = _linear_schedule(
        noise_parameter[0], noise_parameter[1], 0, max_iterations
    )

    cnt_record = 0
    show_col_name = True  # Only for a pretty display of the logs
    # Slightly faster to create the list once outside of the loop
    all_parameters = [x for x in frame_encoder.parameters()]

    for cnt in range(max_iterations):
        # print(sum(v.abs().sum() for _, v in best_model.items()))

        # ------- Patience mechanism
        if cnt - cnt_record > patience:
            if cosine_scheduling_lr:
                # reload the best model so far
                frame_encoder.set_param(best_model)
                optimizer.load_state_dict(best_optimizer_state)

                current_lr = learning_rate_scheduler.state_dict()["_last_lr"][0]
                # actualise the best optimizer lr with current lr
                for g in optimizer.param_groups:
                    g["lr"] = current_lr

                cnt_record = cnt
            else:
                # exceeding the patience level ends the phase
                break

        # ------- Actual optimization
        # This is slightly faster than optimizer.zero_grad()
        for param in all_parameters:
            param.grad = None

        # forward / backward
        out_forward = frame_encoder.forward(
            reference_frames=raw_references,
            quantizer_noise_type=quantizer_noise_type,
            quantizer_type=quantizer_type,
            soft_round_temperature=cur_softround_temperature,
            noise_parameter=cur_noise_parameter,
        )

        loss_function_output = loss_function(
            out_forward.decoded_image,
            out_forward.rate,
            frame.data.data,
            lmbda=frame_encoder_manager.lmbda,
            rate_mlp_bit=0.0,
            compute_logs=False,
        )
        loss_function_output.loss.backward()
        clip_grad_norm_(all_parameters, 1e-1, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        frame_encoder_manager.iterations_counter += 1

        # ------- Validation
        # Each freq_valid iteration or at the end of the phase, compute validation loss and log stuff
        if ((cnt + 1) % frequency_validation == 0) or (cnt + 1 == max_iterations):
            #  a. Update iterations counter and training time and test model
            frame_encoder_manager.total_training_time_sec += time.time() - start_time
            start_time = time.time()

            # b. Test the model and check whether we've beaten our record
            encoder_logs = test(frame_encoder, frame, frame_encoder_manager)

            flag_new_record = False

            if encoder_logs.loss < encoder_logs_best.loss:
                # A record must have at least -0.001 bpp or + 0.001 dB. A smaller improvement
                # does not matter.
                delta_psnr = encoder_logs.psnr_db - encoder_logs_best.psnr_db
                delta_bpp = (
                    encoder_logs.rate_latent_bpp - encoder_logs_best.rate_latent_bpp
                )
                flag_new_record = delta_bpp < 0.001 or delta_psnr > 0.001

            if flag_new_record:
                # Save best model
                best_model = frame_encoder.get_param()
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())

                # ========================= reporting ========================= #
                this_phase_psnr_gain = (
                    encoder_logs.psnr_db - initial_encoder_logs.psnr_db
                )
                this_phase_bpp_gain = (
                    encoder_logs.rate_latent_bpp - initial_encoder_logs.rate_latent_bpp
                )

                log_new_record = ""
                log_new_record += f"{this_phase_bpp_gain:+6.3f} bpp "
                log_new_record += f"{this_phase_psnr_gain:+6.3f} db"
                # ========================= reporting ========================= #

                # Update new record
                encoder_logs_best = encoder_logs
                cnt_record = cnt
            else:
                log_new_record = ""

            # Show column name a single time
            additional_data = {
                "lr": f"{start_lr if not cosine_scheduling_lr else learning_rate_scheduler.get_last_lr()[0]:.8f}",
                "optim": ",".join(optimized_module),
                "patience": (patience - cnt + cnt_record) // frequency_validation,
                "q_type": f"{quantizer_type:12s}",
                "sr_temp": f"{cur_softround_temperature:.5f}",
                "n_type": f"{quantizer_noise_type:12s}",
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

            # Update soft rounding temperature and noise_parameter
            cur_softround_temperature = _linear_schedule(
                softround_temperature[0],
                softround_temperature[1],
                cnt,
                max_iterations,
            )
            cur_noise_parameter = _linear_schedule(
                noise_parameter[0],
                noise_parameter[1],
                cnt,
                max_iterations,
            )

            if cosine_scheduling_lr:
                learning_rate_scheduler.step()

            frame_encoder.set_to_train()

    # At the end of the training, we load the best model
    frame_encoder.set_param(best_model)
    return frame_encoder
