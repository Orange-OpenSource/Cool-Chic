# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import argparse
import os
from typing import Any, Dict, List


# ----- Arguments related to Cool-chic parameters
def _parse_synthesis_layers(layers_synthesis: str) -> List[str]:
    """The layers of the synthesis are presented in as a coma-separated string.
    This simply splits up the different substrings and return them.

    Args:
        layers_synthesis (str): Command line argument for the synthesis.

    Returns:
        List[str]: List of string where the i-th element described the i-th
            synthesis layer
    """
    parsed_layer_synth = [x for x in layers_synthesis.split(",") if x != ""]

    assert parsed_layer_synth, (
        "Synthesis should have at least one layer, found nothing. \n"
        f"--layers_synthesis={layers_synthesis} does not work!\n"
        "Try something like 32-1-linear-relu,X-1-linear-none,"
        "X-3-residual-relu,X-3-residual-none"
    )

    return parsed_layer_synth


def _parse_arm_archi(arm: str) -> Dict[str, int]:
    """The arm is described as <dim_arm>,<n_hidden_layers_arm>.
    Split up this string to return the value as a dict.

    Args:
        arm (str): Command line argument for the ARM.

    Returns:
        Dict[str, int]: The ARM architecture
    """
    assert len(arm.split(",")) == 2, f"--arm format should be X,Y." f" Found {arm}"

    dim_arm, n_hidden_layers_arm = [int(x) for x in arm.split(",")]
    arm_param = {"dim_arm": dim_arm, "n_hidden_layers_arm": n_hidden_layers_arm}
    return arm_param


def _parse_n_ft_per_res(n_ft_per_res: str) -> List[int]:
    """The number of feature per resolution is a coma-separated string.
    This simply splits up the different substrings and return them.

    Args:
        n_ft_per_res (str): Something like "1,1,1,1,1,1,1" for 7 latent grids
        with different resolution and 1 feature each.

    Returns:
        List[int]: The i-th element is the number of features for the i-th
        latent, i.e. the latent of a resolution (H / 2^i, W / 2^i).
    """

    n_ft_per_res = [int(x) for x in n_ft_per_res.split(",") if x != ""]

    for x in n_ft_per_res:
        assert x in [0, 1], (
            f"--n_ft_per_res should only contains 0 or 1. Found {n_ft_per_res}"
        )

    return n_ft_per_res


def get_coolchic_param_from_args(
    args: argparse.Namespace,
    coolchic_enc_name: str,
) -> Dict[str, Any]:
    layers_synthesis = _parse_synthesis_layers(
        getattr(args, f"layers_synthesis_{coolchic_enc_name}")
    )
    n_ft_per_res = _parse_n_ft_per_res(
        getattr(args, f"n_ft_per_res_{coolchic_enc_name}")
    )


    coolchic_param = {
        "layers_synthesis": layers_synthesis,
        "n_ft_per_res": n_ft_per_res,
        "ups_k_size": getattr(args, f"ups_k_size_{coolchic_enc_name}"),
        "ups_preconcat_k_size": getattr(
            args, f"ups_preconcat_k_size_{coolchic_enc_name}"
        ),
    }

    # Common randomness only for the residue if we want to optimize
    # the wasserstein distance.
    if coolchic_enc_name == "residue" and args.tune == "wasserstein":
        n_ft_per_res_cr = [1, 1, 1, 1, 1, 1, 1]
        
        assert len(n_ft_per_res_cr) <= len(n_ft_per_res), (
            "It is not possible to have smallest resolutions for the common "
            "randomness than for the transmitted latent. Found:\n"
            f"--n_ft_per_res_{coolchic_enc_name}={n_ft_per_res}\n"
            f"--n_ft_per_res_cr={n_ft_per_res_cr}\n"
        )

        coolchic_param["n_ft_per_res_cr"] = n_ft_per_res_cr


    # Add ARM parameters
    coolchic_param.update(_parse_arm_archi(getattr(args, f"arm_{coolchic_enc_name}")))

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

    assert (
        n_frames > 0
    ), f"There must be at least one frame to encode. Found --n_frames={n_frames}"

    assert (
        frame_offset >= 0
    ), f"Negative frame_offset is not possible. Found --frame_offset={frame_offset}"

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


# ----- Arguments related to the frame encoder manager i.e. training preset etc.
def get_manager_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Perform some check on the argparse object used to collect the command
    line parameters. Return a dictionary ready to be plugged into the
    ``FrameEncoderManager`` constructor.

    Args:
        args (argparse.Namespace): Command-line argument parser.

    Returns:
        Dict[str, Any]: Dictionary ready to be plugged into the
            ``FrameEncoderManager`` constructor.
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
        dist_weight = {"mse": 0.2, "wasserstein": 0.8}


    else:
        raise argparse.ArgumentTypeError(f"Unknown --tune. Found {args.tune}")

    frame_encoder_manager = {
        "preset_name": args.preset,
        "start_lr": args.start_lr,
        "lmbda": args.lmbda,
        "n_loops": args.n_train_loops,
        "n_itr": args.n_itr,
        "n_itr_pretrain_motion": args.n_itr_pretrain_motion,
        "dist_weight": dist_weight,
    }

    return frame_encoder_manager

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

