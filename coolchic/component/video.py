# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import copy
import os
from typing import Dict, List

import torch
from torch import Tensor

from coolchic.component.core.coolchic import CoolChicEncoder, CoolChicEncoderParameter
from coolchic.component.frame import NAME_COOLCHIC_ENC, FrameEncoder
from coolchic.component.intercoding.globalmotion import get_global_translation
from coolchic.component.intercoding.raft import get_raft_optical_flow
from coolchic.component.intercoding.warp import WarpParameter
from coolchic.io.framedata import FrameData
from coolchic.io.io import load_frame_data_from_file, save_frame_data_to_file
from coolchic.nnquant.quantizemodel import quantize_model
from coolchic.training.presets import Preset
from coolchic.training.test import test
from coolchic.training.train import train
from coolchic.training.warmup import warmup
from coolchic.utils.codingstructure import CodingStructure, Frame


def encode_one_frame(
    video_path: str,
    coding_structure: CodingStructure,
    coolchic_enc_param: Dict[NAME_COOLCHIC_ENC, CoolChicEncoderParameter],
    warp_parameter: WarpParameter,
    coding_index: int,
    training_preset: Preset,
    device: torch.device = "cpu",
) -> None:
    frame = coding_structure.get_frame_from_coding_order(coding_index)

    def change_n_out_synth(layers_synth: List[str], n_out: int) -> List[str]:
        """Change the number of output features in the list of strings
        describing the synthesis architecture. It replaces "X" with n_out. E.g.

        From [8-1-linear-relu,X-1-linear-none,X-3-residual-none]
        To   [8-1-linear-relu,2-1-linear-none,2-3-residual-none]

        If n_out = 2

        Args:
            layers_synth (List[str]): List of strings describing the different
                synthesis layers
            n_out (int): Number of desired output.

        Returns:
            List[str]: List of strings with the proper number of output features.
        """
        return [lay.replace("X", str(n_out)) for lay in layers_synth]

    print(
        "Frame being encoded\n"
        "-------------------\n\n"
        f"{frame.pretty_string(show_header=True, show_bottom_line=True)}"
    )

    # Load the original data and its references
    frame.set_frame_data(
        load_frame_data_from_file(video_path, frame.display_order + frame.frame_offset)
    )
    frame.set_refs_data(get_ref_data(frame, coding_structure))
    prefix_save = f"{_get_frame_path_prefix(frame.display_order)}"

    if frame.frame_type == "I" and "motion" in coolchic_enc_param:
        coolchic_enc_param.pop("motion")

    # Override default enc_param settings
    for cc_enc_name in coolchic_enc_param:
        coolchic_enc_param[cc_enc_name].encoder_gain = 16

        img_min_max = None
        if frame.frame_type != "I":
            continue

        # Frame I specific
        if training_preset.lmbda < 0.0002:
            coolchic_enc_param[cc_enc_name].encoder_gain = 24
        elif training_preset.lmbda < 0.0005:
            coolchic_enc_param[cc_enc_name].encoder_gain = 20

        # YUV420 or YUV444 specific
        if frame.data.frame_data_type == "yuv420":
            img_min = torch.tensor([c.min().item() for c in frame.data.data.values()]).view(-1, 1)
            img_max = torch.tensor([c.max().item() for c in frame.data.data.values()]).view(-1, 1)
            img_min_max = torch.cat((img_min, img_max), dim=1)
        elif frame.data.frame_data_type == "yuv444":
            img_min = frame.data.data.amin(dim=(-1, -2), keepdim=True).view(-1, 1)
            img_max = frame.data.data.amax(dim=(-1, -2), keepdim=True).view(-1, 1)
            img_min_max = torch.cat((img_min, img_max), dim=1)

    # Determine the number of output features for all cc_enc
    n_ft_out_cc_enc = {}
    match frame.frame_type:
        case "I":
            n_ft_out_cc_enc["residue"] = 3
        case "P":
            n_ft_out_cc_enc["residue"] = 4
            n_ft_out_cc_enc["motion"] = 2
        case "B":
            n_ft_out_cc_enc["residue"] = 5
            n_ft_out_cc_enc["motion"] = 4

    # Change number of output features for the synthesis of all cc_enc
    for cc_enc_name in coolchic_enc_param:
        coolchic_enc_param[cc_enc_name].layers_synthesis = change_n_out_synth(
            coolchic_enc_param[cc_enc_name].layers_synthesis, n_ft_out_cc_enc[cc_enc_name]
        )

    # Log a few details about the model
    print(f"{training_preset.pretty_string()}")

    tmp_str = "Decoder architectures\n"
    tmp_str += "---------------------\n\n"
    for name, cc_enc_param in coolchic_enc_param.items():
        title = f"{name} decoder parameters\n{'-' * len(name)}-------------------\n"
        tmp_str += title
        tmp_str += cc_enc_param.pretty_string() + "\n"

    if frame.frame_type != "I":
        title = "Warper parameters\n-----------------\n"
        tmp_str += title
        tmp_str += warp_parameter.pretty_string() + "\n"

    print(tmp_str)

    frame = frame_to_device(frame, device)

    # Motion pre-training guided with raft.
    if frame.frame_type != "I":
        # We need to output the shifted refs so that raft estimates the motion
        # between the frame to code and the reference shifted by the global flow.
        shifted_ref_data, global_flows = get_global_translation(frame.data, frame.refs_data)

        # Since we are not an I-frame, we have at least 1 ref.
        # For simplicity add an all-zero global flow to this list
        # It does not really matter since there is only one ref, but the
        # set_global_flow() function expects two global flows
        if len(global_flows) == 1:
            global_flows.append(torch.zeros_like(global_flows[0]))

        # Get the RAFT-estimated optical flows
        raft_flows = get_raft_optical_flow(frame.data, shifted_ref_data)

        # Specific gain parameter to learn the optical flow as an image

        pre_trained_motion_enc_param = copy.deepcopy(coolchic_enc_param.get("motion"))
        pre_trained_motion_enc_param.encoder_gain = 16

        pretrained_motion_enc = guided_motion_pretraining(
            list_target_flow=raft_flows,
            motion_enc_param=pre_trained_motion_enc_param,
            device=device,
            training_preset=training_preset,
        )

        # For the remainder of the training we'll need to scale the latent
        # to adapt for a different encoder gains.
        gain_ratio = (
            coolchic_enc_param.get("motion").encoder_gain
            / pre_trained_motion_enc_param.encoder_gain
        )

        for idx_lat, latent_i in enumerate(pretrained_motion_enc.latent_grids):
            pretrained_motion_enc.latent_grids[idx_lat] = latent_i * gain_ratio

    # Get the number of candidates from the initial warm-up phase
    n_candidates = training_preset.warmup.phases[0].candidates

    list_candidates = []
    for idx_candidate in range(n_candidates):
        cur_frame_encoder = FrameEncoder(
            coolchic_enc_param=coolchic_enc_param,
            warp_parameter=warp_parameter,
            frame_type=frame.frame_type,
            frame_data_type=frame.data.frame_data_type,
            bitdepth=frame.data.bitdepth,
            index_references=frame.index_references,
            frame_display_index=frame.display_order,
        )

        cur_frame_encoder.coolchic_enc["residue"].synthesis.init_output_transform(img_min_max)

        # Plug inside the pretrained motion cool-chic encoder
        if frame.frame_type != "I":
            if n_candidates < 2:
                print(
                    "There is a single warm-up candidate which is based on "
                    "a RAFT-guided pre-training."
                )

            # Load the pre-trained results only for half the candidate!
            if idx_candidate % 2:
                # print(f"\nUsing the pre-trained motion cool-chic for warm-up candidate {idx_candidate}")
                cur_frame_encoder.coolchic_enc["motion"].set_param(
                    pretrained_motion_enc.get_param()
                )

            cur_frame_encoder.set_global_flow(global_flows[0], global_flows[1])

        list_candidates.append(cur_frame_encoder)

    # Use the first candidate of the list to log the architecture
    with open(f"{prefix_save}archi.txt", "w") as f_out:
        for cc_name, cc_enc in list_candidates[0].coolchic_enc.items():
            f_out.write(cc_name + "\n\n" + str(cc_enc) + "\n\n")
            f_out.write(cc_enc.str_complexity() + "\n")

    print(f"{list_candidates[0].pretty_string()}\n\n")

    # Use warm-up to find the best initialization among the list
    # of candidates parameters.
    frame_encoder = warmup(
        list_candidates=list_candidates,
        training_preset=training_preset,
        frame=frame,
        device=device,
    )

    frame_encoder.to_device(device)

    # Compile only after the warm-up to compile only once.
    # No compilation for torch version anterior to 2.5.0
    major, minor = [int(x) for x in torch.__version__.split(".")[:2]]
    use_compile = False
    if major > 2:
        use_compile = True
    elif major == 2:
        use_compile = minor >= 5

    if training_preset.preset_name == "debug":
        print("Skip compilation when debugging\n")
    elif training_preset._get_total_training_iterations(what="main") <= 1000:
        print("No compilation when training has less than 1000 iterations")
    elif not use_compile:
        print("No compilation for torch version anterior to 2.5.0\n")
    elif not torch.cuda.is_available():
        print("No compilation when no cuda is available")
    else:
        print("Compiling frame encoder!\n")
        torch._dynamo.reset()
        torch.set_float32_matmul_precision("high")
        frame_encoder = torch.compile(
            frame_encoder,
            dynamic=False,
            mode="reduce-overhead",
            # Some part of the frame_encoder forward (420-related stuff)
            # are not (yet) compatible with compilation. So we can't
            # capture the full graph for yuv420 frame
            fullgraph=frame.data.frame_data_type != "yuv420",
            # disable=True,
        )

    for idx_phase, training_phase in enumerate(training_preset.training_phases):
        print(f"{'-' * 30} Training phase: {idx_phase:>2} {'-' * 30}\n")
        frame_encoder = train(
            frame_encoder=frame_encoder,
            frame=frame,
            training_phase=training_phase,
        )

        if training_phase.quantize_model:
            # Store full precision parameters inside the
            # frame_encoder for later use if needed
            frame_encoder._store_full_precision_param()
            frame_encoder = quantize_model(
                frame_encoder=frame_encoder,
                frame=frame,
                dist_weight=training_phase.dist_weight,
                lmbda=training_phase.lmbda,
            )

        phase_results = test(
            frame_encoder=frame_encoder,
            frame=frame,
            dist_weight=training_phase.dist_weight,
            lmbda=training_phase.lmbda,
        )

        print("\nResults at the end of the phase:")
        print("--------------------------------")
        print(f"\n{phase_results.pretty_string(show_col_name=True, mode='short')}\n")

    # compute the final loss
    final_results = test(
        frame_encoder=frame_encoder,
        frame=frame,
        dist_weight=training_preset.dist_weight,
        lmbda=training_preset.lmbda,
    )

    # Save results
    with open(f"{prefix_save}results_encoder.tsv", "w") as f_out:
        f_out.write(final_results.pretty_string(show_col_name=True, mode="all") + "\n")

    frame_encoder.save(f"{prefix_save}frame_encoder.pt")

    # Save decoded image
    frame_encoder.set_to_eval()
    frame_encoder_output = frame_encoder.forward(
        reference_frames=[ref_i.data for ref_i in frame.refs_data],
        quantizer_noise_type="none",
        quantizer_type="hardround",
        AC_MAX_VAL=-1,
        flag_additional_outputs=True,
    )

    dec_path = f"{prefix_save}decoded-{frame.seq_name}"
    dec_img = frame_encoder_output.decoded_image

    if frame.data.frame_data_type == "rgb":
        extension = ".png" if frame.data.bitdepth == 8 else ".ppm"
    # YUV
    else:
        extension = ".yuv"

    save_frame_data_to_file(
        FrameData(
            bitdepth=frame.data.bitdepth, frame_data_type=frame.data.frame_data_type, data=dec_img
        ),
        file_path=dec_path + extension,
    )
    print("End of training \n\n")


def get_ref_data(
    frame: Frame,
    coding_structure: CodingStructure,
) -> List[FrameData]:
    """Return a list of the (decoded) reference frames. The decoded data
    are obtained by recursively inferring the already learned FrameEncoder.

    Args:
        frame: The frame whose reference(s) we want.

    Returns:
        The decoded reference frames.
    """
    all_ref_data = []

    for ref_display_idx in frame.index_references:
        ref_frame = coding_structure.get_frame_from_display_order(ref_display_idx)

        # TODO: code duplication here
        # For now, reference are necessary a .yuv file
        dec_ref_path = f"{_get_frame_path_prefix(ref_display_idx)}decoded-{frame.seq_name}.yuv"

        assert os.path.isfile(dec_ref_path), (
            f"Cannot find the decoded reference {dec_ref_path}.\n"
            "Hint: make sure that you have already encoded the frame "
            f"{ref_frame.frame_type}{ref_frame.display_order} using "
            f"--coding_idx={ref_frame.coding_order}"
        )

        # ! idx_display_order = 0 because this is a single-frame video
        # ! which directly corresponds to the reference frame.
        ref_data = load_frame_data_from_file(dec_ref_path, idx_display_order=0)

        # Load the references from the already stored
        ref_frame.set_frame_data(ref_data)

        all_ref_data.append(ref_frame.data)

    return all_ref_data


def _get_frame_path_prefix(frame_display_order: int) -> str:
    """Return a string with a prefix that should be appended to every file
    linked to a frame.

    Args:
        frame_display_order: Display order of the frame

    Returns:
        The prefix
    """
    return str(frame_display_order).zfill(4) + "-"


def guided_motion_pretraining(
    list_target_flow: List[Tensor],
    motion_enc_param: CoolChicEncoderParameter,
    training_preset: Preset,
    device: torch.device,
) -> CoolChicEncoder:
    """Learn a CoolChicEncoder imitating one or several optical flows
    list_target_flow as if it were an image, i.e. minimizing the MSE
    between the target flow(s) and the output flow(s) of a CoolChicEncoder.

    The resulting CoolChicEncoder is re-used as initialisation for the motion
    CoolChicEncoder to encode a Frame.

    Args:
        list_target_flow: List of N (= 1 or = 2) flows with shape [1, 2, H, W]
            to be imitated.
        motion_enc_param: Parameters (architecture etc.) for the motion encoder.
        of_encoder_manager: Parameters (training, schedule etc.) for the motion
            pre training.
        device: Device on which the motion pre-training will be run.

    Returns:
        CoolChicEncoder: Trained CoolChicEncoder, imitating the optical flow(s).
    """

    print("\nMotion pre-training")
    print("----------------------")

    assert len(list_target_flow) in [1, 2], (
        "guided_motion_pretraining expects one or two optical flows to be "
        f"pre-trained. Found {len(list_target_flow)} optical flows."
    )

    # From a list of N [1, 2, H, W] flows to a [1, 2N, H, W] dense
    # tensor to be used as a reference for the training.
    target_flow = torch.cat(list_target_flow, dim=1)

    # Create a FrameEncoder to code these optical flows as an intra frame
    # For this we have to wrap this target_flow inside a frame object.
    # Use max bitdepth possible. flow just means dense outputs + no
    # clamping in [0., 1.]
    target_flow_frame = Frame(
        coding_order=0,
        display_order=0,
        data=FrameData(bitdepth=8, frame_data_type="flow", data=target_flow),
    )

    # Learn the optical flows as a normal image, with an encoder gain of 16
    frame_encoder_motion = FrameEncoder(
        # A single "residue" (i.e. all intra) encoder to learn this motion
        coolchic_enc_param={"residue": motion_enc_param},
        warp_parameter=None,
        frame_type=target_flow_frame.frame_type,
        frame_data_type=target_flow_frame.data.frame_data_type,
        bitdepth=target_flow_frame.data.bitdepth,
        index_references=[],
        frame_display_index=0,
    )
    frame_encoder_motion.to_device(device)
    target_flow_frame = frame_to_device(target_flow_frame, device)

    for _, training_phase in enumerate(training_preset.motion_pretrain_phase):
        frame_encoder_motion = train(
            frame_encoder=frame_encoder_motion,
            frame=target_flow_frame,
            training_phase=training_phase,
        )

    # Return the trained frame_encoder
    # It is stored as the "residue" since it is learned as an image
    return frame_encoder_motion.coolchic_enc["residue"]


def frame_to_device(frame: Frame, device: torch.device) -> Frame:
    """
    Push a frame and its reference to a device

    Args:
        frame: Frame to be pushed to a device
        device: Required device

    Returns:
        Frame: Frame pushed to a device
    """

    if frame.data is not None:
        frame_data_to_device(frame.data, device)

    for index_ref in range(len(frame.refs_data)):
        if frame.refs_data[index_ref] is not None:
            frame_data_to_device(frame.refs_data[index_ref], device)

    return frame


def frame_data_to_device(frame_data: FrameData, device: torch.device) -> FrameData:
    """Push the data attribute to the relevant device **in place**.

    Args:
        frame_data: The data to be pushed
        device: The device on which the frame data should be pushed.

    Returns:
        FrameData: The data on the correct device

    """
    from coolchic.io.format.yuv import yuv_dict_to_device

    if frame_data.frame_data_type == "yuv420":
        frame_data.data = yuv_dict_to_device(frame_data.data, device)
    else:
        frame_data.data = frame_data.data.to(device)

    return frame_data
