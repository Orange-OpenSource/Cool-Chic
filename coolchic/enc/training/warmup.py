# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import copy
import time
from typing import List

from enc.training.manager import FrameEncoderManager
from enc.component.frame import FrameEncoder
from enc.training.test import test
from enc.training.train import train
from enc.utils.codingstructure import Frame
from enc.utils.device import POSSIBLE_DEVICE
from enc.utils.misc import mem_info


def warmup(
    frame_encoder_manager: FrameEncoderManager,
    list_candidates: List[FrameEncoder],
    frame: Frame,
    device: POSSIBLE_DEVICE,
) -> FrameEncoder:
    """Perform the warm-up for a frame encoder. It consists in multiple stages
    with several candidates, filtering out the best N candidates at each stage.
    For instance, we can start with 8 different FrameEncoder. We train each of
    them for 400 iterations. Then we keep the best 4 of them for 400 additional
    iterations, while finally keeping the final best one.

    .. warning::

        The parameter ``frame_encoder_manager`` tracking the encoding time of
        the frame (``total_training_time_sec``) and the number of encoding
        iterations (``iterations_counter``) is modified** in place** by this
        function.

    Args:
        frame_encoder_manager: Contains (among other things) the rate
            constraint :math:`\\lambda` and description of the
            warm-up preset. It is also used to track the total encoding time
            and encoding iterations. Modified in place.
        list_candidates: The different candidates among which the warm-up will
            find the best starting point.
        frame: The original image to be compressed and its references.
        device: On which device should the training run.

    Returns:
        Warmuped frame encoder, with a great initialization.
    """

    start_time = time.time()
    warmup = frame_encoder_manager.preset.warmup

    _col_width = 14

    # Construct the list of candidates. Each of them has its own parameters,
    # unique ID and metrics (not yet evaluated so it is set to None).
    all_candidates = [
        {"metrics": None, "id": id_candidate, "encoder": param_candidate}
        for id_candidate, param_candidate in enumerate(list_candidates)
    ]

    for idx_warmup_phase, warmup_phase in enumerate(warmup.phases):
        print(f'{"-" * 30}  Warm-up phase: {idx_warmup_phase:>2} {"-" * 30}')

        mem_info(f"Warmup-{idx_warmup_phase:02d}")

        # At the beginning of the all warm-up phases except the first one,
        # keep the desired number of best candidates.
        if idx_warmup_phase != 0:
            n_elements_to_remove = len(all_candidates) - warmup_phase.candidates
            for _ in range(n_elements_to_remove):
                all_candidates.pop()
            # all_candidates = all_candidates[: warmup_phase.candidates]

        # i is just the index of A candidate in the all_candidates
        # list. It is **not** a unique identifier for this candidate. This is
        # given by:
        #   all_candidates[i].get('id')
        # The all_candidates list gives the ordered list of the best performing
        # models so its order may change.

        # # Check that we do have different candidates with different parameters
        # print('------\nbefore')
        # for x in all_candidates:
        #     print(f"{x.get('id')}   {sum([v.abs().sum() for k, v in x.get('param').items() if 'synthesis' in k])}")

        # Train all (remaining) candidates for a little bit
        for i in range(warmup_phase.candidates):
            cur_candidate = all_candidates[i]
            cur_id = cur_candidate.get("id")
            frame_encoder = cur_candidate.get("encoder")
            frame_encoder.to_device(device)

            print(
                f"\nCandidate n° {i:<2}, ID = {cur_id:<2}:"
                + "\n-------------------------\n"
            )
            mem_info(f"Warmup-cand-in {idx_warmup_phase:02d}-{i:02d}")

            frame_encoder = train(
                frame_encoder=frame_encoder,
                frame=frame,
                frame_encoder_manager=frame_encoder_manager,
                start_lr=warmup_phase.training_phase.lr,
                lmbda=frame_encoder_manager.lmbda,
                cosine_scheduling_lr=warmup_phase.training_phase.schedule_lr,
                max_iterations=warmup_phase.training_phase.max_itr,
                patience=warmup_phase.training_phase.patience,
                frequency_validation=warmup_phase.training_phase.freq_valid,
                optimized_module=warmup_phase.training_phase.optimized_module,
                quantizer_type=warmup_phase.training_phase.quantizer_type,
                quantizer_noise_type=warmup_phase.training_phase.quantizer_noise_type,
                softround_temperature=warmup_phase.training_phase.softround_temperature,
                noise_parameter=warmup_phase.training_phase.noise_parameter
            )

            metrics = test(frame_encoder, frame, frame_encoder_manager)
            frame_encoder.to_device("cpu")

            # Put the updated candidate back into the list.
            cur_candidate["encoder"] = frame_encoder
            cur_candidate["metrics"] = metrics
            all_candidates[i] = cur_candidate

        all_candidates = sorted(all_candidates, key=lambda x: x.get("metrics").loss)

        # # Check that we do have different candidates with different parameters
        # for x in all_candidates:
        #     print(f"{x.get('id')}   {sum([v.abs().sum() for k, v in x.get('encoder').get_param().items() if 'synthesis' in k])}")
        # print('after\n------')

        # Print the results of this warm-up phase
        s = "\n\nPerformance at the end of the warm-up phase:\n\n"
        s += f'{"ID":^{6}}|{"loss":^{_col_width}}|{"rate_bpp":^{_col_width}}|{"psnr_db":^{_col_width}}|\n'
        s += f'------|{"-" * _col_width}|{"-" * _col_width}|{"-" * _col_width}|\n'
        for candidate in all_candidates:
            s += f'{candidate.get("id"):^{6}}|'
            s += f'{candidate.get("metrics").loss.item() * 1e3:^{_col_width}.4f}|'
            s += f'{candidate.get("metrics").total_rate_latent_bpp:^{_col_width}.4f}|'
            s += f'{candidate.get("metrics").psnr_db:^{_col_width}.4f}|'
            s += "\n"
        print(s)

    # Keep only the best model
    frame_encoder = copy.deepcopy(all_candidates[0].get("encoder"))

    # We've already worked for that many second during warm up
    warmup_duration = time.time() - start_time

    print("Intra Warm-up is done!")
    print(f"Intra Warm-up time [s]: {warmup_duration:.2f}")
    print(f'Intra Winner ID       : {all_candidates[0].get("id")}\n')

    return frame_encoder
