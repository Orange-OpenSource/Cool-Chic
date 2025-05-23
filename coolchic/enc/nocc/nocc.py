# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from dataclasses import dataclass

import torch
from enc.component.coolchic import CoolChicEncoder, CoolChicEncoderParameter
from enc.nocc.analysis import Analysis


@dataclass
class NoCoolChic:
    cc_enc: CoolChicEncoder
    analysis: Analysis


def load_no_coolchic(path_file: str) -> NoCoolChic:
    loaded_data = torch.load(path_file, map_location="cpu", weights_only=False)

    coolchic_enc = CoolChicEncoder(loaded_data["cc_enc_param"])
    coolchic_enc.set_param(loaded_data["cc_enc"])

    analysis = Analysis(
        n_feat=loaded_data["analysis_n_feat"], n_blocks=loaded_data["analysis_n_blocks"]
    )
    analysis.load_state_dict(loaded_data["analysis"])

    no_cc = NoCoolChic(cc_enc=coolchic_enc, analysis=analysis)
    return no_cc


def check_match_archi(
    nocc_param: CoolChicEncoderParameter,
    cc_param: CoolChicEncoderParameter,
    error_on_mismatch: bool = True,
) -> bool:
    """Check if the decoder described by two ``CoolChicEncoderParameter`` matches.
    This is used when initiating a decoder from a pre-trained No-Cool-Chic.
    Ideally we would want the architecture to be equal (ARM, Upsampling and
    synthesis). This function returns True if it matches, False otherwise.

    Args:
        nocc_param (CoolChicEncoderParameter): No-Cool-chic architecture (source).
        cc_param (CoolChicEncoderParameter): Cool-chic architecture (target.)
        error_on_mismatch (bool, optional): If True, raise an error
            in case of mismatch. Defaults to True.

    Raises:
        ValueError: If ``error_on_mismatch`` an error can be raised when the
            architectures do not match.

    Returns:
        bool: True if the architecture match, False otherwise.
    """
    match = True
    msg = ""

    # ARM
    if (cc_param.dim_arm != nocc_param.dim_arm) or (
        cc_param.n_hidden_layers_arm != nocc_param.n_hidden_layers_arm
    ):
        msg += (
            f"ARM in No-cool-chic "
            "is different from the one asked in command line. Use --arm_residue="
            f"{nocc_param.dim_arm},{nocc_param.n_hidden_layers_arm}."
        )
        match = False

    # Synthesis. We necessarily have 3 outputs with no-coolchic. (RGB)
    n_out = 3
    cc_synth = [
        lay.replace("X", str(n_out)) for lay in cc_param.layers_synthesis
    ]
    if cc_synth != nocc_param.layers_synthesis:
        msg += (
            f"Synthesis in No-cool-chic "
            "is different from the one asked in command line. Use "
            f"--layers_synthesis_residue={','.join(nocc_param.layers_synthesis)}"
        )
        match = False

    # Latent
    if cc_param.n_ft_per_res != nocc_param.n_ft_per_res:
        msg += (
            f"Latent domain in No-cool-chic "
            "is different from the one asked in command line. Use "
            f"--n_ft_per_res_residue={','.join(nocc_param.n_ft_per_res)}"
        )
        match = False

    # Upsampling
    if cc_param.ups_k_size != nocc_param.ups_k_size:
        msg += (
            f"Upsampling kernel size in No-cool-chic "
            "is different from the one asked in command line. Use "
            f"--ups_k_size_residue={','.join(nocc_param.ups_k_size)}"
        )
        match = False

    if cc_param.ups_preconcat_k_size != nocc_param.ups_preconcat_k_size:
        msg += (
            f"Pre-concat upsampling kernel size in No-cool-chic  "
            "is different from the one asked in command line. Use "
            f"--ups_preconcat_k_size_residue={','.join(nocc_param.ups_preconcat_k_size)}"
        )
        match = False

    if error_on_mismatch and not match:
        raise ValueError(
            "Cannot use No-Cool-chic. There is at least one architecture "
            "mismatch between No-Cool-chic architecture"
            "and the requested Cool-chic architecture."
            f"\n\n{msg}\n\n"
            "Otherwise, use error_on_mismatch=False in check_match_archi() "
            "to simply ignore init from No-Cool-chic when architecture mismatch"
        )

    return match
