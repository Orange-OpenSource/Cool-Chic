# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

from coolchic.io.io import load_frame_data_from_file


# ----- Arguments related to Cool-chic parameters
def _parse_synthesis_layers(layers_synthesis: str) -> Optional[Tuple[List[str], bool]]:
    """The layers of the synthesis are presented in as a coma-separated string.
    This simply splits up the different substrings and return them.

    Args:
        layers_synthesis (str): Command line argument for the synthesis.

    Returns:
        Optional[Tuple[List[str], bool]]: List of string where the i-th element
            described the i-th synthesis layer of the main branch. The boolean
            value indicate wether a stabiliser layer is used.
    """
    # Empty string is set to None
    if layers_synthesis == "":
        return None

    flag_stabiliser = layers_synthesis.endswith("/stabiliser")
    layers_synthesis = layers_synthesis.replace("/stabiliser", "")

    parsed_layer_synth = [x for x in layers_synthesis.split(",") if x != ""]

    assert parsed_layer_synth, (
        "Synthesis should have at least one layer, found nothing. \n"
        f"--layers_synthesis={layers_synthesis} does not work!\n"
        "Try something like 32-1-linear-relu,X-1-linear-none,"
        "X-3-residual-relu,X-3-residual-none"
    )

    return parsed_layer_synth, flag_stabiliser


def _parse_spatial_arm_archi(spatial_arm: str) -> Dict[str, int]:
    """The spatial arm is described as <n_spatial_context>,<n_hidden_layers_arm>.
    Split up this string to return the value as a dict.

    Args:
        spatial_arm (str): Command line argument for the ARM.

    Returns:
        Dict[str, int]: The ARM architecture
    """

    flag_stabiliser = spatial_arm.endswith("/stabiliser")
    spatial_arm = spatial_arm.replace("/stabiliser", "")

    assert len(spatial_arm.split(",")) == 2, f"--arm format should be X,Y. Found {spatial_arm}"

    spatial_context_arm, n_hidden_layers_arm = [int(x) for x in spatial_arm.split(",")]
    arm_param = {
        "spatial_context_arm": spatial_context_arm,
        "n_hidden_layers_arm": n_hidden_layers_arm,
        "linear_stabiliser_arm": flag_stabiliser,
    }
    return arm_param


def _parse_output_feature_ifce(output_feature_ifce: int) -> int:
    x = int(output_feature_ifce)

    assert x >= 0, f"--output_feature_ifce should be a positive number. Found {x}"

    return x


def _parse_latent_resolution(latent_resolution: str, n_pixels: int) -> Tuple[int, int]:

    # Adaptive lowest resolution: 2^-6 = 1/64 for images with less than 1 million
    # pixels, 1/128 for images with 1 to 3 millions pixel and 1/256 for the bigger images
    if latent_resolution == "auto":
        if n_pixels < 1_000_000:
            v = 6
        elif n_pixels < 3_000_000:
            v = 7
        else:
            v = 8
        res = (0, v)
    else:
        res = tuple([int(x) for x in latent_resolution.split("-") if x != ""])

    return res


def _parse_hyperlatent_resolution(
    hyperlatent_resolution: str, n_pixels: int
) -> Optional[Tuple[int, int]]:
    if hyperlatent_resolution == "no":
        return None
    elif hyperlatent_resolution == "auto":
        # Adaptive lowest resolution: 2^-6 = 1/64 for images with less than 1
        # million pixels, 1/128 for images with 1 to 3 millions pixel and 1/256
        # for the bigger images.
        if n_pixels < 1_000_000:
            v = 6
        elif n_pixels < 3_000_000:
            v = 7
        else:
            v = 8
        return (4, v)
    else:
        return tuple([int(x) for x in hyperlatent_resolution.split("-") if x != ""])


def _parse_ifce_resolution(ifce_resolution: str) -> Optional[Tuple[int, int]]:
    if ifce_resolution == "no":
        return None
    else:
        return tuple([int(x) for x in ifce_resolution.split("-") if x != ""])


def get_coolchic_param_from_args(
    args: argparse.Namespace,
    coolchic_enc_name: str,
) -> Dict[str, Any]:

    # Load the first image of the video to get the number of pixels
    frame = load_frame_data_from_file(args.input, idx_display_order=0)
    if frame.frame_data_type in ["rgb", "yuv444"]:
        height, width = frame.data.size()[-2:]
    elif frame.frame_data_type in ["yuv420"]:
        height, width = frame.data.get("y").size()[-2:]

    n_pixels = height * width

    layers_synthesis, linear_stabiliser_synth = _parse_synthesis_layers(
        getattr(args, f"layers_synthesis_{coolchic_enc_name}")
    )

    output_feature_ifce = _parse_output_feature_ifce(
        getattr(args, f"output_feature_ifce_{coolchic_enc_name}")
    )

    latent_resolution = _parse_latent_resolution(
        getattr(args, f"latent_resolution_{coolchic_enc_name}"), n_pixels
    )

    hyperlatent_resolution = _parse_hyperlatent_resolution(
        getattr(args, f"hyperlatent_resolution_{coolchic_enc_name}"), n_pixels
    )

    ifce_resolution = _parse_ifce_resolution(getattr(args, f"ifce_resolution_{coolchic_enc_name}"))
    if ifce_resolution is None and output_feature_ifce != 0:
        print(
            f"--ifce_resolution_{coolchic_enc_name}=no --> Setting --output_feature_ifce_"
            f"{coolchic_enc_name} to 0."
        )
        output_feature_ifce = 0

    # Common randomness only for the residue if we want to optimize
    # the wasserstein distance.
    if coolchic_enc_name == "residue" and args.tune == "wasserstein":
        flag_common_randomness = True
    else:
        flag_common_randomness = False

    # Motion **must** use nearest neighbor upsampling to have blocks of BxB pixels with
    # the exact same motion vector
    if coolchic_enc_name == "motion":
        final_upsampling_type = "nearest"
    elif coolchic_enc_name == "residue":
        final_upsampling_type = "bicubic"

    coolchic_param = {
        "layers_synthesis": layers_synthesis,
        "linear_stabiliser_synth": linear_stabiliser_synth,
        "latent_resolution": latent_resolution,
        "ups_k_size": getattr(args, f"ups_k_size_{coolchic_enc_name}"),
        "ups_preconcat_k_size": getattr(args, f"ups_preconcat_k_size_{coolchic_enc_name}"),
        "output_feature_ifce": output_feature_ifce,
        "hyperlatent_resolution": hyperlatent_resolution,
        "ifce_resolution": ifce_resolution,
        "img_size": (height, width),
        "flag_common_randomness": flag_common_randomness,
        "final_upsampling_type": final_upsampling_type,
    }

    # Add ARM parameters
    coolchic_param.update(_parse_spatial_arm_archi(getattr(args, f"arm_{coolchic_enc_name}")))

    return coolchic_param


# ----- Arguments related to the coding structure
def _is_image(file_path: str) -> bool:
    """Return True is file extension is an image extension ie JPEG, PNG or PPM.

    Args:
        file_path (str): Path of the file.

    Returns:
        bool: True is file is an "image".
    """

    possible_file_extension = ["png", "jpeg", "jpg", "ppm"]

    for ext in possible_file_extension:
        if file_path.endswith(f".{ext}"):
            return True

        if file_path.endswith(f".{ext.capitalize()}"):
            return True

    return False


def _parse_frame_pos(frame_pos_str: str, n_frames: int) -> List[int]:
    """Parse the command line arguments for --intra_pos or --p_pos.

    Format:
    -------

        Format is 0,4,7 if you want the frame 0, 4 and 7 to be intra frames.

        -1 can be used to denote the last frame, -2 the 2nd to last etc.

        x-y is a range from x (included) to y (included). This does not work
        with the negative indexing.

        0,4-7,-2 ==> Intra for the frame 0, 4, 5, 6, 7 and the 2nd to last."

    Args:
        frame_pos_str: Command line arguments.
        n_frames: Number of frames to code. Correspond to --n_frames and
            it is required to handle negative indexing

    Returns:
        List[int]: List of frame positions by display order.
    """
    if frame_pos_str == "":
        return []

    pos = []
    for tmp in frame_pos_str.split(","):
        if tmp == "":
            continue

        # Positive or negative integer
        try:
            tmp = int(tmp)
            if tmp < 0:
                tmp = n_frames - abs(tmp)
            pos.append(tmp)

        # Range x-y
        except ValueError:
            start, end = [int(x) for x in tmp.split("-")]
            for x in range(start, end + 1):
                pos.append(x)

    pos.sort()
    return pos


def get_coding_structure_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Perform some check on the argparse object used to collect the command
    line parameters. Return a dictionary ready to be plugged into the
    ``CodingStructure`` constructor.

    Args:
        args (argparse.Namespace): Command-line argument parser.

    Returns:
        Dict[str, Any]: Dictionary ready to be plugged into the ``CodingStructure``
            constructor.
    """
    n_frames = args.n_frames
    frame_offset = args.frame_offset

    assert n_frames > 0, f"There must be at least one frame to encode. Found --n_frames={n_frames}"

    assert frame_offset >= 0, (
        f"Negative frame_offset is not possible. Found --frame_offset={frame_offset}"
    )

    if "input" in args:
        if _is_image(args.input):
            assert args.n_frames == 1, (
                f"Encoding a PNG, JPEG or PPM image {args.input} must be done with"
                f" --n_frames=1. Found --n_frames = {args.n_frames}"
            )

    coding_structure_config = {
        "n_frames": n_frames,
        "intra_pos": _parse_frame_pos(args.intra_pos, n_frames),
        "p_pos": _parse_frame_pos(args.p_pos, n_frames),
        "seq_name": os.path.basename(args.input).split(".")[0] if "input" in args else "",
        "frame_offset": frame_offset,
    }
    return coding_structure_config


def get_warp_param_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Perform some check on the argparse object used to collect the command
    line parameters. Return a dictionary ready to be plugged into the
    ``WarpParameter`` constructor.

    Args:
        args (argparse.Namespace): Command-line argument parser.

    Returns:
        Dict[str, Any]: Dictionary ready to be plugged into the
            ``WarpParameter`` constructor.
    """
    warp_parameter = {
        "filter_size": args.warp_filter_size,
    }
    return warp_parameter


def get_preset_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Perform some check on the argparse object used to collect the command
    line parameters. Return a dictionary ready to be plugged into the
    ``Preset`` constructor.

    Args:
        args (argparse.Namespace): Command-line argument parser.

    Returns:
        Dict[str, Any]: Dictionary ready to be plugged into the
            ``Preset`` constructor.
    """
    if args.tune == "mse":
        dist_weight = {"mse": 1.0}

    elif args.tune == "wasserstein":
        if args.input.endswith(".yuv"):
            raise argparse.ArgumentTypeError(
                "--tune=wasserstein can not be used with YUV files. Use --tune=mse"
            )

        # Value determined empirically, see
        # "Perceptually optimised Cool-chic for CLIC 2025", Philippe et al.
        dist_weight = {"mse": 0.2, "wasserstein": 0.8 / 200}

    else:
        raise argparse.ArgumentTypeError(f"Unknown --tune. Found {args.tune}")

    preset_parameters = {
        "start_lr": args.start_lr,
        "lmbda": args.lmbda,
        "itr_main_training": args.n_itr,
        "itr_motion_pretrain": args.n_itr_pretrain_motion,
        "dist_weight": dist_weight,
    }

    return preset_parameters
