# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import copy
import os
import subprocess
import time
from typing import Dict, List, Tuple

import torch
from enc.utils.manager import FrameEncoderManager
from enc.component.coolchic import CoolChicEncoderParameter
from enc.component.frame import FrameEncoder, load_frame_encoder
from enc.training.quantizemodel import quantize_model
from enc.training.test import test
from enc.training.train import train
from enc.training.warmup import warmup
from enc.utils.codingstructure import CodingStructure, Frame, FrameData
from enc.utils.misc import POSSIBLE_DEVICE, TrainingExitCode, is_job_over, mem_info
from enc.utils.yuv import load_frame_data_from_file

class VideoEncoder():

    def __init__(
        self,
        coding_structure: CodingStructure,
        shared_coolchic_parameter: CoolChicEncoderParameter,
        shared_frame_encoder_manager: FrameEncoderManager,
    ):
        """A VideoEncoder object is our main object. Its purpose is to encode
        a video i.e. one I-frame followed by 0 to N inter (P or B) frames.

        Args:
            coding_structure: The coding structure (organization of the
                different frames) used to encode the video.
            shared_coolchic_parameter: Common parameters for all Cool-chic of
                all frames (*e.g.* synthesis architecture). Can be overridden
                later in the encode function to better suits the need of each
                individual frame.
            shared_frame_encoder_manager: Common training parameters for all
                frames (*e.g.* max. number of iterations). It can be overridden
                later in the encode function to better suits the need of each
                individual frame.
        """

        self.coding_structure = coding_structure
        self.shared_coolchic_parameter = shared_coolchic_parameter
        self.shared_frame_encoder_manager = shared_frame_encoder_manager

        # This starts empty and is filled during the successive training
        # Dictionary keys are the coding index.
        # For each key, we have the corresponding FrameEncoder and the
        # corresponding FrameEncoderManager.
        self.all_frame_encoders: Dict[
            str, Tuple[FrameEncoder, FrameEncoderManager]
        ] = {}


    def encode(
        self,
        path_original_sequence: str,
        device: POSSIBLE_DEVICE,
        workdir: str,
        job_duration_min: int = -1,
    ) -> TrainingExitCode:
        """Main training function of a ``VideoEncoder``. Encode all required
        frames (*i.e.* as stated in ``self.coding_structure``) of the video
        located at ``path_original_sequence``. This will fill the dictionary
        ``self.all_frame_encoders`` containing the successively overfitted
        frame encoders.

        There is a series of nested loops to encode the video, following
        roughly this process:

        .. code-block:: python

            # Code all frames
            for idx_coding_order in range(n_frames):
                # Perform n_loops independent encoding
                for idx_loop in range(n_loops):
                    # Find the best initialization
                    frame_encoder = warmup(...)
                    # Perform the successive training stages
                    for training_phase in all_training_phases:
                        frame_encoder = train(frame_encoder, training_phase)
                    # Training is over, test and save
                    results = test(frame_encoder)
                    frame_encoder.save()

        Args:
            path_original_sequence: Absolute path to the original image
                or video to be compressed.
            device: On which device should the training run
            workdir: Where we'll save many thing
            job_duration_min: Exit and save the job after
                this duration is passed. Use -1 to only exit at the end of the
                entire encoding. Default to -1.

        Returns:
            Either ``TrainingExitCode.REQUEUE`` if job has run for
            longer than ``job_duration_min`` and should be put back into the job
            queue, or ``TrainingExitCode.END`` if the encoding is actually over.

        """
        start_time = time.time()
        n_frames = self.coding_structure.get_number_of_frames()

        for idx_coding_order in range(n_frames):
            frame = self.coding_structure.get_frame_from_coding_order(idx_coding_order)

            if frame.already_encoded:
                continue

            # Load the original data and its references
            frame.data = load_frame_data_from_file(
                path_original_sequence, frame.display_order
            )
            frame.refs_data = self.get_ref_data(frame)

            # Everything concerning this frame will be written here
            frame_workdir = self.get_frame_workdir(workdir, frame.display_order)

            current_coolchic_parameter = copy.deepcopy(self.shared_coolchic_parameter)
            current_coolchic_parameter.set_image_size(frame.data.img_size)
            current_coolchic_parameter.encoder_gain = (
                16 if frame.frame_type == "I" else 16
            )

            match frame.frame_type:
                case "I":
                    n_output_synthesis = 3
                case "P":
                    n_output_synthesis = 6
                case "B":
                    n_output_synthesis = 9
                case _:
                    print(
                        f"Unknown frame_type {frame.frame_type}"
                    )

            # Change the number of channels for the synthesis output
            current_coolchic_parameter.layers_synthesis = [
                lay.replace("X", str(n_output_synthesis))
                for lay in current_coolchic_parameter.layers_synthesis
            ]


            # We have started to encode this frame so we already have a
            # frame_encoder_manager associated
            if str(idx_coding_order) in self.all_frame_encoders:
                _, frame_encoder_manager = (
                    self.all_frame_encoders.get(str(idx_coding_order))
                )

            # We need to create a new frame_encoder_manager
            else:
                print(
                    "-" * 80 + "\n"
                    + f'{" " * 12} Coding frame {frame.coding_order + 1} / {n_frames} '
                    + f"- Display order: {frame.display_order} - "
                    + f"Coding order: {frame.coding_order}\n"
                    + "-" * 80
                )

                # ----- Set the parameters for the frame
                frame_encoder_manager = copy.deepcopy(
                    self.shared_frame_encoder_manager
                )
                # Change the lambda according to the depth of the frame in the GOP
                # The deeper the frame, the bigger the lambda, the smaller the rate
                frame_encoder_manager.lmbda = self.get_lmbda_from_depth(
                    frame.depth, self.shared_frame_encoder_manager.lmbda
                )

                # Plug the current frame type into the current frame encoder manager
                frame_encoder_manager.frame_type = frame.frame_type


                subprocess.call(f"mkdir -p {frame_workdir}", shell=True)

                # Log a few details about the model
                print(f"\n{frame_encoder_manager.pretty_string()}")
                print(f"{current_coolchic_parameter.pretty_string()}")
                print(f"{frame_encoder_manager.preset.pretty_string()}")


            for index_loop in range(
                frame_encoder_manager.loop_counter,
                frame_encoder_manager.n_loops,
            ):
                print(
                    "-" * 80
                    + "\n"
                    + f'{" " * 30} Training loop {frame_encoder_manager.loop_counter + 1} / '
                    + f"{frame_encoder_manager.n_loops}\n"
                    + "-" * 80
                )

                # Remove the ARM integerization before training, it will
                # be set later by the quantize_model() function
                # frame_encoder.coolchic_encoder.arm.set_quant(0)
                # frame_encoder.to_device(device)
                frame.to_device(device)

                # Get the number of candidates from the initial warm-up phase
                n_initial_warmup_candidate = (
                    frame_encoder_manager.preset.warmup.phases[
                        0
                    ].candidates
                )

                list_candidates = [
                    FrameEncoder(
                        coolchic_encoder_param=current_coolchic_parameter,
                        frame_type=frame.frame_type,
                        frame_data_type=frame.data.frame_data_type,
                        bitdepth=frame.data.bitdepth
                    )
                    for _ in range(n_initial_warmup_candidate)
                ]

                # Use the first candidate of the list to log the architecture
                with open(f"{frame_workdir}/archi.txt", "w") as f_out:
                    f_out.write(str(list_candidates[0].coolchic_encoder) + "\n\n")
                    f_out.write(list_candidates[0].coolchic_encoder.str_complexity() + "\n")

                # Use warm-up to find the best initialization among the list
                # of candidates parameters.
                frame_encoder = warmup(
                    frame_encoder_manager=frame_encoder_manager,
                    list_candidates=list_candidates,
                    frame=frame,
                    device=device,
                )
                frame_encoder.to_device(device)

                for idx_phase, training_phase in enumerate(frame_encoder_manager.preset.all_phases):
                    print(f'{"-" * 30} Training phase: {idx_phase:>2} {"-" * 30}\n')
                    mem_info("Training phase " + str(idx_phase))
                    frame_encoder = train(
                        frame_encoder=frame_encoder,
                        frame=frame,
                        frame_encoder_manager=frame_encoder_manager,
                        start_lr=training_phase.lr,
                        cosine_scheduling_lr=training_phase.schedule_lr,
                        max_iterations=training_phase.max_itr,
                        frequency_validation=training_phase.freq_valid,
                        patience=training_phase.patience,
                        optimized_module=training_phase.optimized_module,
                        quantizer_type=training_phase.quantizer_type,
                        quantizer_noise_type=training_phase.quantizer_noise_type,
                        softround_temperature=training_phase.softround_temperature,
                        noise_parameter=training_phase.noise_parameter,
                    )

                    if training_phase.quantize_model:
                        # Store full precision parameters inside the
                        # frame_encoder for later use if needed
                        frame_encoder.coolchic_encoder._store_full_precision_param()
                        frame_encoder = quantize_model(
                            frame_encoder,
                            frame,
                            frame_encoder_manager,
                        )

                    phase_results = test(
                        frame_encoder,
                        frame,
                        frame_encoder_manager,
                    )

                    print("\nResults at the end of the phase:")
                    print("--------------------------------")
                    print(
                        f'\n{phase_results.pretty_string(show_col_name=True, mode="short")}\n'
                    )

                # At the end of each loop, compute the final loss
                loop_results = test(
                    frame_encoder,
                    frame,
                    frame_encoder_manager,
                )

                # Write results file
                path_results_log = f"{frame_workdir}results_loop_{frame_encoder_manager.loop_counter + 1}.tsv"
                with open(path_results_log, "w") as f_out:
                    f_out.write(
                        loop_results.pretty_string(show_col_name=True, mode="all") + "\n"
                    )

                # We've beaten our record
                if frame_encoder_manager.record_beaten(loop_results.loss):
                    print(f'Best loss beaten at loop {frame_encoder_manager.loop_counter + 1}')
                    print(f'Previous best loss: {frame_encoder_manager.best_loss * 1e3 :.6f}')
                    print(f'New best loss     : {loop_results.loss.cpu().item() * 1e3 :.6f}')

                    frame_encoder_manager.set_best_loss(loop_results.loss.cpu().item())

                    # Save best results
                    with open(f'{frame_workdir}results_best.tsv', 'w') as f_out:
                        f_out.write(loop_results.pretty_string(show_col_name=True, mode='all') + '\n')
                    self.concat_results_file(workdir)

                    best_frame_encoder = frame_encoder

                # We haven't beaten our record, keep the old frame encoder as
                # the current best frame encoder
                else:
                    best_frame_encoder = self.all_frame_encoders[str(frame.coding_order)][0]

                frame_encoder_manager.loop_counter += 1

                # Store the current best FrameEncoder and the corresponding
                # frame_encoder_manager
                self.all_frame_encoders[str(frame.coding_order)] = (
                    copy.deepcopy(best_frame_encoder),
                    copy.deepcopy(frame_encoder_manager)
                )

                print('End of training loop\n\n')

                self.save(f'{workdir}video_encoder.pt')
                # The save function unload the decoded frames and the original
                # ones. We need to reload them
                frame.data = load_frame_data_from_file(
                    path_original_sequence, frame.display_order
                )
                frame.refs_data = self.get_ref_data(frame)

                if is_job_over(start_time=start_time, max_duration_job_min=job_duration_min):
                    return TrainingExitCode.REQUEUE

            self.coding_structure.set_encoded_flag(
                coding_order=frame.coding_order, flag_value=True
            )
            print(self.coding_structure.pretty_string())
            self.save(f'{workdir}video_encoder.pt')

        return TrainingExitCode.END

    def get_frame_workdir(self, workdir: str, frame_display_order: int) -> str:
        """Compute the absolute path for the workdir of one frame.

        Args:
            workdir: Main working directory of the video encoder
            frame_display_order: Display order of the frame

        Returns:
            Working directory of the frame
        """
        return f"{workdir}/frame_{str(frame_display_order).zfill(3)}/"

    def concat_results_file(self, workdir: str) -> None:
        """Look at all the already encoded frames inside ``workdir`` and
        concatenate their result files (``workdir/frame_XXX/results_best.tsv``)
        into a single result file ``workdir/results_best.tsv``.

        Args:
            workdir: Working directory of the video encoder
        """
        list_results_file = []
        for idx_display_order in range(self.coding_structure.get_number_of_frames()):
            cur_res_file = (
                self.get_frame_workdir(workdir, idx_display_order) + "results_best.tsv"
            )
            if not os.path.isfile(cur_res_file):
                continue

            list_results_file.append(cur_res_file)

        # decoded_frame_name is something like decoded_416x240_1p_yuv420_8b.yuv
        out_path = workdir + "results_best.tsv"

        subprocess.call(f"rm -f {out_path}", shell=True)
        for idx, frame_path in enumerate(list_results_file):
            if idx == 0:
                subprocess.call(f"cat {frame_path} >> {out_path}", shell=True)
            # Print only the second line (no need for the column name)
            else:
                subprocess.call(
                    f"cat {frame_path} | head -2 | tail -1 >> {out_path}", shell=True
                )

    @torch.no_grad()
    def get_ref_data(self, frame: Frame) -> List[FrameData]:
        """Return a list of the (decoded) reference frames. The decoded data
        are obtained by recursively inferring the already learned FrameEncoder.

        Args:
            frame: The frame whose reference(s) we want.

        Returns:
            The decoded reference frames.
        """

        # We obtain the reference frames by re-inferring the already encoded frames.
        ref_data = []

        # idx_ref is in display order
        for idx_ref in frame.index_references:
            ref_frame = self.coding_structure.get_frame_from_display_order(idx_ref)

            # No need to re-infer the reference, this has already been decoded
            if ref_frame.decoded_data is not None:
                pass
            else:
                ref_frame.refs_data = self.get_ref_data(ref_frame)
                print(
                    f"get_ref_data(): Decoding frame {ref_frame.display_order:<3}..."
                )

                # Load the best encoder for the reference frame
                # No need to load the corresponding frame_encoder_manager
                # hence the "_"
                frame_encoder, _ = self.all_frame_encoders.get(str(ref_frame.coding_order))

                # Infer it to get the data of the references
                frame_encoder.set_to_eval()
                frame_encoder.to_device("cpu")

                ref_frame.upsample_reference_to_444()

                frame_encoder_out = frame_encoder.forward(
                    reference_frames=[ref_i.data for ref_i in ref_frame.refs_data],
                    quantizer_noise_type="none",
                    quantizer_type="hardround",
                    AC_MAX_VAL=-1,
                    flag_additional_outputs=False,
                )

                ref_frame.set_decoded_data(
                    FrameData(
                        frame_encoder.bitdepth,
                        frame_encoder.frame_data_type,
                        frame_encoder_out.decoded_image,
                    )
                )

            ref_data.append(ref_frame.decoded_data)

        return ref_data

    def get_lmbda_from_depth(self, depth: float, initial_lmbda: float) -> float:
        """Perform the QP offset as follows:

            .. math::

                \\lambda_i = (\\frac{3}{2})^{d} \\lambda,

        Args:
            depth: The depth :math:`d` of the frame in the GOP.
                See encoder/utils/coding_structure.py for more info
            initial_lmbda: The lmbda of the I frame :math:`\\lambda`.

        Returns:
            The lambda of the current frame :math:`\\lambda_i`.
        """
        return initial_lmbda * (1.5**depth)

    def save(self, save_path: str) -> None:
        """Save current VideoEncoder at given path. It contains everything,
        the ``CodingStructure``, the shared parameters between the different frames
        as well as all the successive ``FrameEncoder`` and their respective
        ``FrameEncoderManager``.

        Args:
            save_path: Where to save the model
        """
        subprocess.call(f"mkdir -p {os.path.dirname(save_path)}", shell=True)

        # We don't need to save the original frames nor the coded ones.
        # The original frames can be reloaded from the dataset. The coded ones
        # can be retrieved by inferring the trained FrameEncoders.
        self.coding_structure.unload_all_original_frames()
        self.coding_structure.unload_all_references_data()
        self.coding_structure.unload_all_decoded_data()

        data_to_save = {
            "coding_structure": self.coding_structure,
            "shared_coolchic_parameter": self.shared_coolchic_parameter,
            "shared_frame_encoder_manager": self.shared_frame_encoder_manager,
            "all_frame_encoders": {},
        }

        for k, v in self.all_frame_encoders.items():
            frame_encoder, frame_encoder_manager = v
            data_to_save["all_frame_encoders"][k] = (frame_encoder.save(), frame_encoder_manager)

        torch.save(data_to_save, save_path)

def load_video_encoder(load_path: str) -> VideoEncoder:
    """Load a video encoder.

    Args:
        load_path: Absolute path where the VideoEncoder should be loaded.

    Returns:
        The loaded VideoEncoder
    """
    print(f"Loading a video encoder from {load_path}")

    raw_data = torch.load(load_path, map_location="cpu")

    # Calling the VideoEncoder constructor automatically reload the
    # original frames.
    video_encoder = VideoEncoder(
        coding_structure=raw_data["coding_structure"],
        shared_coolchic_parameter=raw_data["shared_coolchic_parameter"],
        shared_frame_encoder_manager=raw_data["shared_frame_encoder_manager"],
    )

    # Load all the frame encoders to reconstruct the reference frames when needed
    # TODO: Load only the required frame encoder
    for k, v in raw_data["all_frame_encoders"].items():
        raw_bytes_frame_encoder, frame_encoder_manager = v
        video_encoder.all_frame_encoders[k] = (
            load_frame_encoder(raw_bytes_frame_encoder),
            frame_encoder_manager,
        )

    return video_encoder
