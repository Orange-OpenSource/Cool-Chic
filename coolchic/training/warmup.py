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

import torch

from coolchic.component.frame import EncoderMonitor, FrameEncoder
from coolchic.training.presets import Preset
from coolchic.training.test import test
from coolchic.training.train import train
from coolchic.utils.codingstructure import Frame


def warmup(
    list_candidates: List[FrameEncoder],
    training_preset: Preset,
    frame: Frame,
    device: torch.device,
) -> FrameEncoder:
    """Perform the warm-up for a frame encoder. It consists in multiple stages
    with several candidates, filtering out the best N candidates at each stage.
    For instance, we can start with 8 different FrameEncoder. We train each of
    them for 400 iterations. Then we keep the best 4 of them for 400 additional
    iterations, while finally keeping the final best one.

    Note that the warm-up always use the MSE as distortion metrics, regardless
    of the parameters provided by ``--tune``.

    Args:
        list_candidates: The different candidates among which the warm-up will
            find the best starting point.
        training_preset: Preset containing the warm-up hyperparameters
        frame: The original image to be compressed and its references.
        device: On which device should the training run.

    Returns:
        Warmuped frame encoder, with a great initialization.
    """

    start_time = time.time()
    warmup = training_preset.warmup

    _col_width = 14

    # Construct the list of candidates. Each of them has its own parameters,
    # unique ID and metrics (not yet evaluated so it is set to None).
    all_candidates = [
        {"metrics": None, "id": id_candidate, "encoder": param_candidate}
        for id_candidate, param_candidate in enumerate(list_candidates)
    ]

    # For the warm-up it is easier to trac the time, loss and iterations
    # with an external EncoderMonitor object since we're playing with multiples
    # FrameEncoder
    ext_encoder_monitor = EncoderMonitor()

    for idx_warmup_phase, warmup_phase in enumerate(warmup.phases):
        print(f"{'-' * 30}  Warm-up phase: {idx_warmup_phase:>2} {'-' * 30}")

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
            # Plug the external monitor so that it is incremented (time, itr) in the training function
            frame_encoder.encoder_monitor = ext_encoder_monitor

            print(f"\nCandidate n° {i:<2}, ID = {cur_id:<2}:" + "\n-------------------------\n")

            frame_encoder = train(
                frame_encoder=frame_encoder,
                frame=frame,
                training_phase=warmup_phase.training_phase,
            )

            metrics = test(
                frame_encoder=frame_encoder,
                frame=frame,
                dist_weight=warmup_phase.training_phase.dist_weight,
                lmbda=warmup_phase.training_phase.lmbda,
            )
            frame_encoder.to_device("cpu")

            # Retrieve
            ext_encoder_monitor = frame_encoder.encoder_monitor
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
        s += f"{'ID':^{6}}|{'loss':^{_col_width}}|{'rate_bpp':^{_col_width}}|{'psnr_db':^{_col_width}}|\n"
        s += f"------|{'-' * _col_width}|{'-' * _col_width}|{'-' * _col_width}|\n"
        for candidate in all_candidates:
            s += f"{candidate.get('id'):^{6}}|"
            s += f"{candidate.get('metrics').loss.item() * 1e3:^{_col_width}.4f}|"
            s += f"{candidate.get('metrics').total_rate_latent_bpp:^{_col_width}.4f}|"
            s += f"{candidate.get('metrics').detailed_dist_db.get('psnr_db'):^{_col_width}.4f}|"
            s += "\n"
        print(s)

    # Keep only the best model
    frame_encoder = copy.deepcopy(all_candidates[0].get("encoder"))

    # We've already worked for that many second during warm up
    warmup_duration = time.time() - start_time

    print("Intra Warm-up is done!")
    print(f"Intra Warm-up time [s]: {warmup_duration:.2f}")
    print(f"Intra Winner ID       : {all_candidates[0].get('id')}\n")

    return frame_encoder
