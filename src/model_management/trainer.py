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

from typing import Tuple
from collections import OrderedDict
from torch import nn
from model_management.loss import loss_fn
from model_management.presets import Preset, TrainerPhase
from model_management.tester import CoolChicEncoderLogs, test
from models.cool_chic import CoolChicEncoder, CoolChicParameter, to_device

from models.mlp_coding import greedy_quantization


def one_training_phase(
    model: CoolChicEncoder, trainer_phase: TrainerPhase
    ) -> Tuple[CoolChicEncoder, CoolChicEncoderLogs]:
    """Train an INR codec

    Args:
        model (CoolChicEncoder): INRCodec module already instantiated
        trainer_phase (TrainerPhase): All the training parameters for this learning phase

    Returns:
        Tuple[CoolChicEncoder, CoolChicEncoderLogs]: The trained model and its logs.
    """
    start_time = time.time()

    # The encoder knows the image to encode as well as the rate constraint
    target = model.param.img
    lmbda = model.param.lmbda

    # === Keep track of the best loss and model for *THIS* current phase ==== #
    # Perform a first test to get the current best logs (it includes the loss)
    this_phase_best_results = test(model)
    this_phase_init_results = this_phase_best_results
    this_phase_best_model = OrderedDict(
        (k, v.detach().clone()) for k, v in model.state_dict().items()
    )
    # === Keep track of the best loss and model for *THIS* current phase ==== #

    model = model.train()

    # =============== Build the list of parameters to optimize ============== #
    # Iteratively construct the list of required parameters... This is kind of a
    # strange syntax, which has been found quite empirically
    parameters_to_optimize = []
    if 'arm' in trainer_phase.optimized_module:
        parameters_to_optimize += [*model.arm.parameters()]
    if 'upsampling' in trainer_phase.optimized_module:
        parameters_to_optimize += [*model.upsampling.parameters()]

    if 'synthesis' in trainer_phase.optimized_module:
        parameters_to_optimize += [*model.synthesis.parameters()]

    if 'latent' in trainer_phase.optimized_module:
        parameters_to_optimize += [*model.latent_grids.parameters(), model.log_2_encoder_gains]

    if 'all' in trainer_phase.optimized_module:
        parameters_to_optimize = model.parameters()

    optimizer = torch.optim.Adam(parameters_to_optimize, lr=trainer_phase.lr)
    # =============== Build the list of parameters to optimize ============== #

    cnt_record = 0
    previous_record_rate_latent_bpp = 50
    previous_record_psnr_db = 0

    # phase optimization
    for cnt in range(trainer_phase.max_itr):
        if cnt - cnt_record > trainer_phase.patience:
            break

        # This is slightly faster than optimizer.zero_grad()
        for param in model.parameters():
           param.grad = None

        # forward / backward
        out_forward = model(use_ste_quant=trainer_phase.ste)
        loss, _ = loss_fn(out_forward, target, lmbda, dist_mode=trainer_phase.dist)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1e-1, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        # Each freq_valid iteration or at the end of the phase, compute validation loss and log stuff
        if ((cnt + 1) % trainer_phase.freq_valid == 0) or (cnt + 1 == trainer_phase.max_itr):
            #  a. Update iterations counter and training time and test model
            model.iterations_counter += trainer_phase.freq_valid
            # Update the elapsed time and reset the start_time value
            model.total_training_time_sec += time.time() - start_time
            start_time = time.time()

            results = test(model)

            # b. Store record ----------------------------------------------- #
            flag_new_record = False

            if results.loss < this_phase_best_results.loss:
                # A record must have at least -0.001 bpp or + 0.001 dB. A smaller improvement
                # does not matter.
                delta_psnr = results.psnr_db - previous_record_psnr_db
                delta_bpp  = results.rate_latent_bpp - previous_record_rate_latent_bpp
                if delta_bpp < 0.001 or delta_psnr > 0.001:
                    flag_new_record = True

            if flag_new_record:
                # Save best model
                for k, v in model.state_dict().items():
                    this_phase_best_model[k].copy_(v)

                # ========================= reporting ========================= #
                this_phase_loss_gain =  100 * (results.loss - this_phase_init_results.loss) / this_phase_init_results.loss
                this_phase_psnr_gain =  results.psnr_db - this_phase_init_results.psnr_db
                this_phase_bpp_gain =  results.rate_latent_bpp - this_phase_init_results.rate_latent_bpp

                log_new_record = f'{this_phase_loss_gain:+5.2f}% {this_phase_bpp_gain:+6.3f} bpp {this_phase_psnr_gain:+6.3f} db'
                # ========================= reporting ========================= #

                # Update new record
                this_phase_best_results = results
                previous_record_psnr_db = results.rate_latent_bpp
                previous_record_rate_latent_bpp = results.psnr_db
                cnt_record = cnt

            else:
                log_new_record = ''

            # c. Print some logs -------------------------------------------- #
            more_data_to_log = {
                'STE': trainer_phase.ste,
                'lr': trainer_phase.lr,
                'optim':trainer_phase.optimized_module,
                'patience': (trainer_phase.patience - cnt + cnt_record) // trainer_phase.freq_valid,
                'gains': log_new_record
            }

            print(
                results.to_string(
                    mode='short',
                    additional_data=more_data_to_log,
                    # Print column on first row
                    print_col_name= (cnt + 1 == trainer_phase.freq_valid) or (cnt + 1 == trainer_phase.max_itr)
                )
            )

            # Restore training mode
            model = model.train()

    # Load best model found for this encoding loop
    model.load_state_dict(this_phase_best_model)

    # Quantize the model parameters at the end of the training phase
    if trainer_phase.quantize_model:
        model = greedy_quantization(model)
    results = test(model)

    return model, results


def do_warmup(cool_chic_param: CoolChicParameter, encoder_preset: Preset, device: str = 'cpu') -> CoolChicEncoder:
    """Perform the warm-up stage i.e. a competition between different models in order to find the
    best starting point.

    Args:
        cool_chic_param (CoolChicParameter): Model hyper parameters
        encoder_preset (Preset): Encoder policy, describes (among other things) the
            warm-up settings.
        device (str, Optional): Either "cuda:0" or "cpu"

    Returns:
        CoolChicEncoder: The warm-up winner i.e. the best starting point
    """
    start_time = time.time()

    print(f'Number of warm-up iterations: {encoder_preset.get_total_warmup_iterations()}')

    _col_width = 14

    cnt_warmup_itr = 0
    for idx_warmup_phase, warmup_phase in enumerate(encoder_preset.all_warmups):
        msg_start_end_phase = f'{"#" * 40}    Warm-up phase: {idx_warmup_phase:>2}    {"#" * 40}'
        print('\n' + msg_start_end_phase)

        # At the beginning of the first warmup phase, we must initialize all the models
        if idx_warmup_phase == 0:
            all_candidates = [
                {'model': CoolChicEncoder(cool_chic_param), 'metrics': None, 'id': idx_model}
                for idx_model in range(warmup_phase.candidates)
            ]
        # At the beginning of the other warm-up phases, keep the desired number of best candidates
        else:
            all_candidates = all_candidates[:warmup_phase.candidates]

        # Construct the training phase object describing the options of this particular
        # warm-up phase
        training_phase = TrainerPhase(lr=warmup_phase.lr, max_itr=warmup_phase.iterations)

        # ! idx_candidate is just the index of one candidate in the all_candidates list. It is **not** a
        # ! unique identifier for this candidate. This is given by:
        # !         all_candidates[idx_candidate].get('id')
        # ! the all_candidates list gives the ordered list of the best performing models so its order may change.
        for idx_candidate, candidate in enumerate(all_candidates):
            print(f'\nCandidate n° {idx_candidate:<2}, ID = {candidate.get("id"):<2}:')
            print(f'-------------------------\n')

            # Send the old model to the required device for training
            old_model = to_device(candidate.get('model'), device)
            # Take into account the number of iterations and time spent by other candidates
            old_model.iterations_counter = cnt_warmup_itr
            old_model.total_training_time_sec = time.time() - start_time

            # Perform one training phase to update the model and the results
            updated_model, updated_metrics = one_training_phase(old_model, training_phase)
            cnt_warmup_itr += training_phase.max_itr


            # Bring back the updated model on cpu. This leaves only one model at a time
            # on the GPU.
            #updated_model = to_device(updated_model, 'cpu')

            # Replace the updated model in the model list
            all_candidates[idx_candidate] = {
                'model': updated_model, 'metrics': updated_metrics, 'id': candidate.get('id')
            }

        # Sort all the models by ascending loss. The best one is all_candidates[0]
        all_candidates = sorted(all_candidates, key=lambda x: x.get('metrics').loss)

        # Print the results of this warm-up phase
        s = f'\n\n'
        s += f'{"ID":^{6}}|{"loss":^{_col_width}}|{"rate_bpp":^{_col_width}}|{"psnr_db":^{_col_width}}|\n'
        for candidate in all_candidates:
            s += f'{candidate.get("id"):^{6}}|'
            s += f'{candidate.get("metrics").loss * 1e3:^{_col_width}.4f}|'
            s += f'{candidate.get("metrics").rate_bpp:^{_col_width}.4f}|'
            s += f'{candidate.get("metrics").psnr_db:^{_col_width}.4f}|'
            s += '\n'
        s += f'\n{msg_start_end_phase}\n'
        print(s)

    # Keep only the best model
    best_model = all_candidates[0].get('model')
    # We've already done that many iterations during warm-up
    best_model.iterations_counter = encoder_preset.get_total_warmup_iterations()

    # We've already worked for that many second during warm up
    warmup_duration =  time.time() - start_time
    best_model.total_training_time_sec = warmup_duration

    print(f'\nWarm-up time [s]: {warmup_duration:.2f}')
    print(f'Winner ID       : {all_candidates[0].get("id")}\n')

    return best_model
