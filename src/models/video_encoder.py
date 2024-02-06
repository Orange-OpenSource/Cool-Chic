# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import copy
import glob
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

import torch
from torch import nn

from encoding_management.coding_structure import FRAME_DATA_TYPE, POSSIBLE_BITDEPTH, CodingStructure, Frame, FrameData
from utils.yuv import convert_420_to_444, load_frame_data_from_file
from models.coolchic_encoder import CoolChicEncoderParameter
from models.frame_encoder import FrameEncoder, FrameEncoderManager, load_frame_encoder
from utils.misc import POSSIBLE_DEVICE, TrainingExitCode



@dataclass
class VideoEncoder(nn.Module):
    # ----- Data for the videos to encode and coding structure
    coding_structure: CodingStructure   # Describe coding / display order & contains the data (pixel values)

    # Contains architecture and internal CoolChic parameters which are share between frames
    shared_coolchic_parameter: CoolChicEncoderParameter

    # Contains training parameter shared between frames
    shared_frame_encoder_manager: FrameEncoderManager

    # Path to the original (i.e. non compressed) sequence
    path_original_sequence: str

    # ==================== Not set by the init function ===================== #
    # This starts empty and is filled during the successive training
    # All the frame encoders, keys are the coding index followed by the training loop index
    # e.g. <codingIndex>_<loopIndex>. Note that loopIndex can be either 0, 1, 2... or
    # be 'best' i.e. 2_best is the key related to the best FrameEncoder for the frame whose
    # coding order is 2
    all_frame_encoders: Dict[str, FrameEncoder] = field(init=False, default_factory=lambda: {})

    # Information on the frames coded, set in the load_data_and_refs() function
    img_size: Tuple[int, int] = field(init=False)
    bitdepth: POSSIBLE_BITDEPTH = field(init=False)
    frame_data_type: FRAME_DATA_TYPE = field(init=False)
    # ==================== Not set by the init function ===================== #

    def train(self, device: POSSIBLE_DEVICE, workdir: str, job_duration_min: int = -1):
        """Main training function of a FrameEncoder. It requires a frame_encoder_save_path
        in order to save the encoder periodically to allow for checkpoint.

        Args:
            device (POSSIBLE_DEVICE): On which device should the training run
            workdir (str): Where we'll save many thing
            job_duration_min (Optional, float): Exit and save the job after this duration
                is passed. Use -1 to only exit at the end of the entire encoding.
                Default to -1.
        """

        start_time = time.time()
        n_frames = self.coding_structure.get_number_of_frames()

        for idx_coding_order in range(n_frames):
            frame = self.coding_structure.get_frame_from_coding_order(idx_coding_order)
            if frame.already_encoded:
                continue

            frame_workdir = self.get_frame_workdir(workdir, frame.display_order)
            subprocess.call(f'mkdir -p {frame_workdir}', shell=True)

            msg = '-' * 80 + '\n'
            msg += f'{" " * 12} Coding frame {frame.coding_order + 1} / {n_frames} - Display order: {frame.display_order} - Coding order: {frame.coding_order}\n'
            msg += '-' * 80
            print(msg)

            # ----- Load the original frame & the references
            frame = self.load_data_and_refs(frame)

            # Keys of the all_frame_encoders dict are as follows:
            # <idx_coding_order>_<idx_loop> where idx_loop can be an integer OR 'best'
            # Here we want to keep the keys whose <idx_coding_order> corresponds to
            # the actual frame we're coding. We also filter out the idx_loop 'best'
            # through the is_digit() check.
            filtered_keys = [
                x.split('_')[1] for x in self.all_frame_encoders
                if x.startswith(f'{idx_coding_order}_') and x.split('_')[-1].isdigit()
            ]

            # Find the element with the largest integer suffix
            last_loop_index = max(filtered_keys, key=lambda x: int(x), default=None)

            # Nothing found, create a new FrameEncoder
            if last_loop_index is None:
                # ----- Set the parameters for the frame
                current_frame_encoder_manager = copy.deepcopy(self.shared_frame_encoder_manager)
                current_coolchic_parameter = copy.deepcopy(self.shared_coolchic_parameter)
                current_coolchic_parameter.img_size = frame.data.img_size
                current_coolchic_parameter.encoder_gain = 16 if frame.frame_type == "I" else 1

                # Change the lambda according to the depth of the frame in the GOP
                # The deeper the frame, the bigger the lambda, the smaller the rate
                current_frame_encoder_manager.lmbda = self.shared_frame_encoder_manager.lmbda * (1.5 ** frame.depth)

                if frame.frame_type == 'I':
                    n_output_synthesis = 3
                elif frame.frame_type == 'P':
                    n_output_synthesis = 6
                elif frame.frame_type == 'B':
                    n_output_synthesis = 9

                # Change the output size of the synthesis
                current_coolchic_parameter.layers_synthesis = [
                    lay.replace('X', str(n_output_synthesis))
                    for lay in current_coolchic_parameter.layers_synthesis
                ]

                frame_encoder = FrameEncoder(
                    frame=frame,
                    coolchic_encoder_param=current_coolchic_parameter,
                    frame_encoder_manager=current_frame_encoder_manager,
                )

                print(f'\n{frame_encoder.frame_encoder_manager.pretty_string()}')
                print(f'{frame_encoder.coolchic_encoder_param.pretty_string()}')
                print(f'{frame_encoder.frame_encoder_manager.preset.pretty_string()}')

                # Log a few details about the model
                with open(f'{frame_workdir}/archi.txt', 'w') as f_out:
                    f_out.write(str(frame_encoder.coolchic_encoder) + '\n\n')
                    f_out.write(frame_encoder.coolchic_encoder.str_complexity() + '\n')

            # Load the corresponding FrameEncoder
            else:
                frame_encoder = self.all_frame_encoders[self.get_key_all_frame_encoders(idx_coding_order, last_loop_index)]

            # Encode the frame
            for index_loop in range(frame_encoder.frame_encoder_manager.loop_counter, frame_encoder.frame_encoder_manager.n_loops):
                training_exit_code = frame_encoder.one_training_loop(
                    device=device,
                    frame_workdir=frame_workdir,
                    path_original_sequence=self.path_original_sequence,
                    start_time=start_time,
                    job_duration_min=job_duration_min
                )

                # Concatenate the different results file
                self.concat_results_file(workdir)

                # Store the current frame encoder to pursue training further
                self.all_frame_encoders[self.get_key_all_frame_encoders(idx_coding_order, index_loop)] = copy.deepcopy(frame_encoder)

                # We've just finished a new loop, save the frame encoder as the best one if we've beaten our record
                if training_exit_code == TrainingExitCode.END and index_loop == frame_encoder.frame_encoder_manager.idx_best_loop:
                    print(f'Storing the frame encoder obtained at loop {index_loop + 1}\n')
                    self.all_frame_encoders[self.get_key_all_frame_encoders(idx_coding_order, 'best')] = copy.deepcopy(frame_encoder)

                # Training is over. Either because it is actually finished or because the
                # allocated time is over. In both case, we're going to save the model.
                self.save(f'{workdir}video_encoder.pt')
                # ! Save remove the data and the ref, we need to reload them!
                frame = self.load_data_and_refs(frame)

                # We want to requeue the training... exit!
                if training_exit_code == TrainingExitCode.REQUEUE:
                    # Training is over. Either because it is actually finished or because the
                    # allocated time is over. In both case, we're going to save the model.
                    # self.save(f'{workdir}video_encoder.pt')
                    sys.exit(TrainingExitCode.REQUEUE.value)


            # Store the encoded frame into the coding structure. For that we retrieve the best loop,
            # infer it to obtain the decoded frame and then we store it inside the coding struct.
            best_frame_encoder = self.all_frame_encoders[self.get_key_all_frame_encoders(idx_coding_order, 'best')]
            best_frame_encoder.to_device(device)

            frame_encoder_logs = best_frame_encoder.test()
            # Indicate that the frame is coded and store the decoded image inside the coding structure
            # object for further usage.
            self.coding_structure.set_encoded_flag(
                idx_coding_order,
                flag_value=True,
                decoded_frame=FrameData(
                    bitdepth=frame.data.bitdepth,
                    frame_data_type=frame.data.frame_data_type,
                    data=frame_encoder_logs.frame_encoder_output.decoded_image
                )
            )

            # Once we're done coding the current frame, we can delete all the frame
            # encoders except the best one
            for index_loop in range(frame_encoder.frame_encoder_manager.n_loops):
                self.all_frame_encoders.pop(
                    self.get_key_all_frame_encoders(idx_coding_order, index_loop)
                )

            # Training is over for this frame. Save the model!
            self.save(f'{workdir}video_encoder.pt')

    def get_key_all_frame_encoders(self, idx_coding_order: int, idx_loop: Union[int, str]) -> str:
        """Construct a key as follows: <idx_coding_order>_<idx_loop>. These are the keys
        indexing the video_encoder.all_frame_encoders dictionary

        Args:
            idx_coding_order (int): Index of the coding order of the frame
            idx_loop (Union[int, str]): Either the training loop index or simply 'best'
                to indicate the best loop.

        Returns:
            str: The key required to index the video_encoder.all_frame_encoders dictionary
        """
        return f'{idx_coding_order}_{idx_loop}'

    def get_frame_workdir(self, workdir: str, frame_display_order: int) -> str:
        """Compute the absolute path for the workdir of one frame.

        Args:
            workdir (str): Main working directory of the video encoder
            frame_display_order (int): Display order of the frame

        Returns:
            str: Working directory of the frame
        """
        return f'{workdir}/frame_{str(frame_display_order).zfill(3)}/'

    def save(self, save_path: str):
        """Save current VideoEncoder at given path. It contains everything,
        the coding structure, the shared parameters between the different frames
        as well as all the successive frame encoder

        Args:
            save_path (str): Where to save the model
        """
        subprocess.call(f'mkdir -p {os.path.dirname(save_path)}', shell=True)

        # We don't need to save the original frames nor the coded ones.
        # The original frames can be reloaded from the dataset. The coded ones
        # can be retrieved by inferring the trained FrameEncoders.
        self.coding_structure.unload_all_original_frames()
        self.coding_structure.unload_all_references_data()
        self.coding_structure.unload_all_decoded_data()

        data_to_save = {
            'coding_structure': self.coding_structure,
            'shared_coolchic_parameter': self.shared_coolchic_parameter,
            'shared_frame_encoder_manager': self.shared_frame_encoder_manager,
            'path_original_sequence': self.path_original_sequence,
            'img_size': self.img_size,
            'bitdepth': self.bitdepth,
            'frame_data_type': self.frame_data_type,
            'all_frame_encoders': {}
        }

        for k, frame_encoder in self.all_frame_encoders.items():
            data_to_save['all_frame_encoders'][k] = frame_encoder.save()

        torch.save(data_to_save, save_path)

    def concat_visualisation(self, workdir: str):
        """Look at all the visualization generated inside workdir and concatenate them to
        obtain a decoded video.

            workdir (str): Working directory of the video encoder
        """
        out_dir = f'{workdir}/visu/'
        subprocess.call(f'mkdir -p {out_dir}', shell=True)

        # During the first loop, delete the old concatenated yuv file
        flag_delete_old_concat = True
        for idx_display_order in range(self.coding_structure.get_number_of_frames()):
            for visu_path in glob.glob(self.get_frame_workdir(workdir, idx_display_order) + 'visu/*.yuv'):
                # visu_name is something like decoded_416x240_1p_yuv420_8b.yuv
                visu_name = os.path.basename(visu_path)

                # decoded_frame_name is something like decoded_416x240_1p_yuv420_8b.yuv
                out_path = out_dir + visu_name

                if flag_delete_old_concat:
                    subprocess.call(f'rm -f {out_path}', shell=True)
                subprocess.call(f'cat {visu_path} >> {out_path}', shell=True)

            # Nothing to delete now
            flag_delete_old_concat = False

    def concat_results_file(self, workdir: str):
        """Look at all the already encoded frames inside workdir and concatenate their
        results_best.tsv into the workdir.

            workdir (str): Working directory of the video encoder
        """
        list_results_file = []
        for idx_display_order in range(self.coding_structure.get_number_of_frames()):
            cur_res_file = self.get_frame_workdir(workdir, idx_display_order) + 'results_best.tsv'
            if not os.path.isfile(cur_res_file):
                continue

            list_results_file.append(cur_res_file)

        # decoded_frame_name is something like decoded_416x240_1p_yuv420_8b.yuv
        out_path = workdir + 'results_best.tsv'

        subprocess.call(f'rm -f {out_path}', shell=True)
        for idx, frame_path in enumerate(list_results_file):
            if idx == 0:
                subprocess.call(f'cat {frame_path} >> {out_path}', shell=True)
            # Print only the second line (no need for the column name)
            else:
                subprocess.call(f'cat {frame_path} | head -2 | tail -1 >> {out_path}', shell=True)

    def generate_all_results_and_visu(self, workdir: str, visu: bool = True):
        """Look at all the already encoded frames inside the working directory and recompute the
        the different results and the visualisations.

        Args:
            workdir (str): Working directory of the video encoder
            visu (bool): Do we generate visu?

        """
        for idx_display_order in range(self.coding_structure.get_number_of_frames()):
            frame = self.coding_structure.get_frame_from_display_order(idx_display_order)

            # Load the original frame & the references
            frame = self.load_data_and_refs(frame)

            if not self.get_key_all_frame_encoders(frame.coding_order, 'best') in self.all_frame_encoders:
                continue

            print(f'generate_all_results_and_visu(): Frame {idx_display_order}')
            # Create subdirectory to output the results and the visualisation
            frame_workdir = self.get_frame_workdir(workdir, idx_display_order)
            subprocess.call(f'mkdir -p {frame_workdir}', shell=True)

            frame_encoder = self.all_frame_encoders[self.get_key_all_frame_encoders(frame.coding_order, 'best')]
            frame_encoder.to_device('cpu')
            frame_encoder_logs = frame_encoder.test()

            # Write results file
            with open(f'{frame_workdir}results_best.tsv', 'w') as f_out:
                f_out.write(frame_encoder_logs.pretty_string(show_col_name=True, mode='all') + '\n')

            # Generate visualisations
            if visu:
                frame_encoder.generate_visualisation(f'{frame_workdir}visu/')
        print('')

        # Concatenate the results and visualisation together
        self.concat_results_file(workdir)
        if visu:
            self.concat_visualisation(workdir)

    @torch.no_grad()
    def load_data_and_refs(self, frame: Frame) -> Frame:
        """Take an "empty" frame (i.e. only the coding order, display order & not
        the original FrameData nor the references data) and fill it with the
        original data and reference data. The original data are obtained by loading
        them from a file. The decoded data are obtained by recursively inferring the
        already learned frame_encoder.

        Args:
            frame (Frame): Kind of empty frame with no FrameData in it.

        Returns:
            Frame: The frame with all the references data and original frame data.
        """
        # --------------------- Load the original data --------------------- #
        frame.data = load_frame_data_from_file(self.path_original_sequence, frame.display_order)
        self.img_size = frame.data.img_size
        self.bitdepth = frame.data.bitdepth
        self.frame_data_type = frame.data.frame_data_type
        # --------------------- Load the original data --------------------- #

        # -------------------- Load the references data -------------------- #
        # We obtain the reference frames by re-inferring the already encoded frames.
        ref_data = []

        # idx_ref is in display order
        for idx_ref in frame.index_references:
            ref_frame = self.coding_structure.get_frame_from_display_order(idx_ref)

            # No need to re-infer the reference, this has already been decoded
            if hasattr(ref_frame, 'decoded_data'):
                pass
            else:
                ref_frame = self.load_data_and_refs(ref_frame)
                print(f'load_data_and_refs(): Decoding frame {ref_frame.display_order:<3}...')

                # Load the best encoder for the reference frame
                frame_encoder = self.all_frame_encoders.get(
                    self.get_key_all_frame_encoders(ref_frame.coding_order, 'best')
                )

                # Infer it to get the data of the references
                frame_encoder.set_to_eval()
                frame_encoder.to_device('cpu')

                # flag_additional_outputs set to True to obtain more output
                frame_encoder_out = frame_encoder.forward(use_ste_quant=False, AC_MAX_VAL=-1, flag_additional_outputs=True)

                # 4:2:0 Reference are upsampled before being used by the InterCodingModule
                if ref_frame.data.frame_data_type == 'yuv420':
                    decoded_frame = convert_420_to_444(frame_encoder_out.decoded_image)
                    ref_frame_data_type = 'yuv444'
                else:
                    decoded_frame = frame_encoder_out.decoded_image
                    ref_frame_data_type = ref_frame.data.frame_data_type

                ref_frame.set_decoded_data(FrameData(ref_frame.data.bitdepth, ref_frame_data_type, decoded_frame))

            ref_data.append(ref_frame.decoded_data)

        frame.set_refs_data(ref_data)
        # -------------------- Load the references data -------------------- #

        return frame

def load_video_encoder(load_path: str) -> VideoEncoder:
    """Load a video encoder located at <load_path>.

    Args:
        load_path (str): Absolute path where the VideoEncoder should be loaded.

    Returns:
        VideoEncoder: The loaded VideoEncoder
    """
    print(f'Loading a video encoder from {load_path}')

    raw_data = torch.load(load_path, map_location='cpu')

    # Calling the VideoEncoder constructor automatically reload the
    # original frames.
    video_encoder = VideoEncoder(
        coding_structure=raw_data['coding_structure'],
        shared_coolchic_parameter=raw_data['shared_coolchic_parameter'],
        shared_frame_encoder_manager=raw_data['shared_frame_encoder_manager'],
        path_original_sequence=raw_data['path_original_sequence'],
    )

    video_encoder.img_size = raw_data['img_size']
    video_encoder.bitdepth = raw_data['bitdepth']
    video_encoder.frame_data_type = raw_data['frame_data_type']

    # Load all the frame encoders to reconstruct the reference frames when needed
    # TODO: Load only the required frame encoder
    for k, raw_bytes in raw_data['all_frame_encoders'].items():
        idx_coding_order = int(k.split('_')[0])
        frame = video_encoder.coding_structure.get_frame_from_coding_order(idx_coding_order)
        video_encoder.all_frame_encoders[k] = load_frame_encoder(raw_bytes, frame=frame)

    return video_encoder