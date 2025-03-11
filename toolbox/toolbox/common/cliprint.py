# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


def dict_to_str(dct: dict, noheader: bool = False) -> str:
    """Transform a dictionary into a string. This is either a 2-line string
    with <keys>\n<val> of if noheader is True, only return a string containing
    the values <val>. Everything is tab separated.

    Args:
        dct (dict): Data to print.
        noheader (bool, optional): Set to true to don't print the keys of
            the dictionary. Defaults to False.

    Returns:
        str: String representing the dictionary
    """
    key = ""
    val = ""
    for k, v in dct.items():
        key = f"{key}\t{k}" if len(key) else k
        val = f"{val}\t{v}" if len(val) else f"{v}"

    final_str = val if noheader else f"{key}\n{val}"
    return final_str
