# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch


def format_results_loss(logs: dict, col_name: bool = False) -> str:
    """Format a dictionary of logs as either a one-row string (if col_name
    is False) or a two-row string (if col_name is True).

    Args:
        logs (dict): output of the loss function containing different metrics.

    Returns:
        str: A one or two-row strings.
    """
    # Log first col name if needed then the values.
    msg = ''
    if col_name:
        for k in logs:
            msg += f'{k:<15}'
        msg += '\n'

    for _, v in logs.items():
        v = v.item() if isinstance(v, torch.Tensor) else v
        v = f'{v}' if isinstance(v, int) else f'{v:5.3f}'
        msg += f'{v:<15}'

    return msg