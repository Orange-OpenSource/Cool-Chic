# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from torch import nn


def pretty_string_ups(upsampling: nn.Module, header: str) -> str:
    """Get a nice string ready to be printed which displays the different layers
    of the upsampling step.

    Something like:

    ..code ::

        header

              +-------------+
        y0 -> | 8x8 TConv2d | -----+
              +-------------+      |
                                   v
              +------------+    +-----+    +-------------+
        y1 -> | 7x7 Conv2d | -> | cat | -> | 8x8 TConv2d | -----+
              +------------+    +-----+    +-------------+      |
                                                                v
                                           +------------+    +-----+    +-------------+
        y2 ------------------------------> | 7x7 Conv2d | -> | cat | -> | 8x8 TConv2d | -> dense
                                           +------------+    +-----+    +-------------+
    Args:
        upsampling (nn.Module): The upsampling module to print
        header (str): A string to append before the layers

    Returns:
        str: A nice string presenting the upsampling.
    """

    lines = []

    assert upsampling.n_ups_kernel == upsampling.n_ups_preconcat_kernel, (
        "Textual representation of the upsampling module excepts to have the"
        "same number of upsampling and pre-concatenation kernels. Found"
        f"n_ups_kernel = {upsampling.n_ups_kernel} and n_ups_preconcat_kernel ="
        f" {upsampling.n_ups_preconcat_kernel}."
    )

    # Useful thing to print
    cat_box = [
        "+-----+",
        "| cat |",
        "+-----+",
    ]
    # ! Warning no space before short_arrow here!
    no_space_short_arrow = "->"
    short_arrow = " -> "

    shorten_names = {
        "ParametrizedUpsamplingSeparableSymmetricConv2d": "Conv2d",
        "ParametrizedUpsamplingSeparableSymmetricConvTranspose2d": "TConv2d",
    }

    lateral_offset = 0

    for idx_ups in range(upsampling.n_ups_kernel + 1):
        latent_name = f"y{idx_ups:<1} "
        mid = f"{latent_name}{'-' * lateral_offset}{no_space_short_arrow} "
        hori_border = ["", ""]

        for i in range(len(hori_border)):
            hori_border[i] += " " * len(mid)

        # First (smallest) latent does not have a pre-concatenation filter
        if idx_ups != 0:
            conv_lay = upsampling.conv2ds[i]
            layer_name = shorten_names.get(
                type(conv_lay).__name__, type(conv_lay).__name__
            )
            k_size = conv_lay.target_k_size
            inside_box = f" {k_size}x{k_size} {layer_name} "
            mid += f"|{inside_box}|{short_arrow}"

            # Add a concatenation box
            for i in range(len(hori_border)):
                hori_border[i] += f"+{'-' * len(inside_box)}+{' ' * len(short_arrow)}"

            mid += cat_box[1] + short_arrow
            hori_border[0] += cat_box[0]
            hori_border[1] += cat_box[2]

            for i in range(len(hori_border)):
                hori_border[i] += f"{' ' * len(short_arrow)}"

            lateral_offset = len(mid) - len(latent_name) - len(no_space_short_arrow) - 1

        tconv_lay = upsampling.conv_transpose2ds[i]
        layer_name = shorten_names.get(
            type(tconv_lay).__name__, type(tconv_lay).__name__
        )
        k_size = tconv_lay.target_k_size
        inside_box = f" {k_size}x{k_size} {layer_name} "
        mid += f"|{inside_box}|"

        for i in range(len(hori_border)):
            hori_border[i] += f"+{'-' * len(inside_box)}+"

        concat_arrow = (
            " "  # Skip one space as in long arrow
            # Account for the long arrow and half of the cat box.
            + "-" * (len(no_space_short_arrow) + len(cat_box[0]) // 2)
        )

        # Last line is a bit specific
        if idx_ups == upsampling.n_ups_kernel:
            mid += f"{short_arrow}dense"
            lines += [hori_border[0], mid, hori_border[1]]

        else:
            mid += concat_arrow + "+"
            hori_border[1] += " " * len(concat_arrow) + "|"
            last_line = " " * (len(mid) - 1) + "v"
            lines += [hori_border[0], mid, hori_border[1], last_line]

    return header + "\n".join(lines)


def pretty_string_nn(
    layers: nn.Sequential, header: str, input_str: str, output_str: str
) -> str:
    """Get a nice string ready to be printed which displays the different layers
    of a neural network.

    Something like:

    ..code ::

        header
                   +--------------------------+
                   |                          |
                   |                          v
                   |   +---------------+    +-----+    +------+      +---------------+
        input_str ---> | Linear 8 -> 8 | -> |  +  | -> | ReLU | ---> | Linear 8 -> 2 | ---> output_str
                       +---------------+    +-----+    +------+      +---------------+

    Args:
        layers (nn.Sequential): The successive layer of the neural network.
        header (str): A string to append before the layers
        input_str (str): A string describing the input of the NN.
        output_str (str): A string describing the output of the NN.

    Returns:
        str: A nice string presenting the neural network architecture.
    """

    lines = [
        "",  # For residual connexions
        "",  # For residual connexions
        "",  # For residual connexions
        "",  # For blocks in themselves
        "",  # For blocks in themselves
        "",  # For blocks in themselves
    ]

    # Name of the layer appears here
    idx_mid_block = len(lines) - 2

    # Horizontal borders above and below the block name
    idx_top_block = len(lines) - 3
    idx_bot_block = len(lines) - 1
    top_bot_blocks = [idx_bot_block, idx_top_block]

    # First three lines are for residual connexions
    res_blocks = list(range(3))

    lines[idx_mid_block] = input_str
    for idx in top_bot_blocks + res_blocks:
        lines[idx] += " " * len(lines[idx_mid_block])

    # Useful thing to print
    plus_box = [
        "+-----+",
        "|  +  |",
        "+-----+",
    ]
    short_arrow = " -> "
    long_arrow = " ---> "

    shorten_names = {"ArmLinear": "Linear", "SynthesisConv2d": "Conv2d"}

    # Print one layer after the other
    for lay in layers:
        is_non_linearity = (
            isinstance(lay, nn.ReLU)
            or isinstance(lay, nn.Identity)
            or isinstance(lay, nn.LeakyReLU)
        )

        arrow_str = short_arrow if is_non_linearity else long_arrow

        layer_name = shorten_names.get(type(lay).__name__, type(lay).__name__)
        if layer_name == "Identity":
            continue

        inside_box = f" {layer_name} "
        if not is_non_linearity:
            inside_box += f"{lay.in_channels} -> {lay.out_channels} "

        if hasattr(lay, "kernel_size"):
            inside_box = f" {lay.kernel_size}x{lay.kernel_size}{inside_box}"

        lines[idx_mid_block] += f"{arrow_str}|{inside_box}|"

        for idx in top_bot_blocks:
            lines[idx] += f"{' ' * len(arrow_str)}+{'-' * len(inside_box)}+"

        is_residual = False
        if hasattr(lay, "residual"):
            is_residual = lay.residual

        if is_residual:
            half_arrow_len = len(arrow_str) // 2
            for idx in res_blocks:
                lines[idx] += " " * half_arrow_len

            res_arrow_len = (
                half_arrow_len
                - 1  # The remaining half of the input arrow, minus 1
                + len(inside_box)
                + 2  # The entire box + 2 for the edges
                + len(short_arrow)  # The short arrow from the layer box to the add box
                + len(plus_box[0]) // 2  # Half of the plus box.
            )
            lines[0] += f"+{'-' * res_arrow_len}+"
            lines[1] += f"|{' ' * res_arrow_len}|"
            lines[2] += f"|{' ' * res_arrow_len}v"

            for idx in res_blocks:
                lines[idx] += " " * (len(plus_box[0]) // 2)

        else:
            for idx in res_blocks:
                lines[idx] += f"{' ' * len(arrow_str)} {' ' * len(inside_box)} "

        if is_residual:
            arrow_str = short_arrow
            lines[idx_mid_block] += f"{arrow_str}{plus_box[1]}"
            for idx in top_bot_blocks:
                lines[idx] += f"{' ' * len(arrow_str)}{plus_box[0]}"

            lines[3] = list(lines[3])
            lines[3][-(res_arrow_len + half_arrow_len + 2)] = "|"
            lines[3] = "".join(lines[3])

    lines[idx_mid_block] += long_arrow + output_str

    return header + "\n".join(lines)
