# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
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
    assert set(n_ft_per_res) == {
        1
    }, f"--n_ft_per_res should only contains 1. Found {n_ft_per_res}"
    return n_ft_per_res


def get_coolchic_param_from_args(args: argparse.Namespace) -> Dict[str, Any]:

    layers_synthesis = _parse_synthesis_layers(getattr(args, "layers_synthesis"))
    n_ft_per_res = _parse_n_ft_per_res(getattr(args, "n_ft_per_res"))

    coolchic_param = {
        "layers_synthesis": layers_synthesis,
        "n_ft_per_res": n_ft_per_res,
        "ups_k_size": getattr(args, "ups_k_size"),
        "ups_preconcat_k_size": getattr(args, "ups_preconcat_k_size"),
    }

    # Add ARM parameters
    coolchic_param.update(_parse_arm_archi(getattr(args, "arm")))

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
    intra_period = args.intra_period
    p_period = args.p_period

    assert intra_period >= 0 and intra_period <= 255, (
        f"Intra period should be in [0, 255]. Found {intra_period}"
    )

    assert p_period >= 0 and p_period <= 255, (
        f"P period should be in [0, 255]. Found {p_period}"
    )

    if _is_image(args.input):
        assert intra_period == 0 and p_period == 0, (
            f"Encoding a PNG, JPEG or PPM image {args.input} must be done with"
            "intra_period = 0 and p_period = 0. Found intra_period = "
            f"{args.intra_period} and p_period = {args.p_period}"
        )

    coding_structure_config = {
        "intra_period": intra_period,
        "p_period": p_period,
        "seq_name": os.path.basename(args.input).split(".")[0],
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
    frame_encoder_manager = {
        "preset_name": args.recipe,
        "start_lr": args.start_lr,
        "lmbda": args.lmbda,
        "n_loops": args.n_train_loops,
        "n_itr": args.n_itr,
    }
    return frame_encoder_manager
