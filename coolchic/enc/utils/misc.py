# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import psutil
import torch


def mem_info(strinfo: str = "Memory allocated") -> None:
    """Convenient printing of the current CPU / GPU memory allocated,
    prefixed by strinfo.

    Args:
        strinfo (str, optional): Printing prefix. Defaults to "Memory allocated".
    """
    mem_cpu = psutil.Process().memory_info().rss
    mem_cpu_GB = mem_cpu / (1024.0 * 1024.0 * 1024.0)

    mem_gpu = torch.cuda.memory_allocated("cuda:0")
    mem_gpu_GB = mem_gpu / (1024.0 * 1024.0 * 1024.0)

    str_display = (
        f"| {strinfo:30s} cpu:{mem_cpu_GB:7.3f} GB --- gpu:{mem_gpu_GB:7.3f} GB |"
    )
    L = len(str_display)
    print(f'{" "*100}{"-"*L}')
    print(f'{" "*100}{str_display}')
    print(f'{" "*100}{"-"*L}')
