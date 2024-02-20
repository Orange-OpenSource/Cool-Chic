# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import struct
import time
import torch

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union, Literal

# ============================ Device management ============================ #
POSSIBLE_DEVICE = Literal['cpu', 'cuda:0']

def get_best_device() -> POSSIBLE_DEVICE:
    """Return the best available device i.e. best ranked one in the following list:
            1. cuda:0
            2. cpu

    Returns:
        POSSIBLE_DEVICE: The best available device
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
# ============================ Device management ============================ #

# =========================== Cluster management ============================ #
# Exiting the program with 42 will requeue the job
class TrainingExitCode(Enum):
    END = 0
    REQUEUE = 42


def is_job_over(start_time: float, max_duration_job_min: int = 45) -> bool:
    """Return true if current time is more than max_duration_job_min after start time.
    Use -1 for max_job_duration_min to always return False


    Args:
        start_time (float): time.time() at the start of the program.
        max_duration_job_min (int): How long we should run. If -1, we never stop
            i.e. we always return False.

    Returns:
        bool: True if current time is more than max_duration_job_min after start time.
    """
    if max_duration_job_min < 0:
        return False

    return (time.time() - start_time) / 60 >= max_duration_job_min
# =========================== Cluster management ============================ #

# ======================= Some useful data structures ======================= #
@dataclass
class DescriptorNN():
    """Contains information (scale, weight, quantization step, ...) about the
    weights and biases of a neural network."""
    weight: Optional[Union[int, float, str]] = None
    bias: Optional[Union[int, float, str]] = None


@dataclass
class DescriptorCoolChic():
    """Contains information about the different sub-networks of Cool-chic."""
    arm: Optional[DescriptorNN] = None
    upsampling: Optional[DescriptorNN] = None
    synthesis: Optional[DescriptorNN] = None
# ======================= Some useful data structures ======================= #

# =============== Neural network quantization & integerization ============== #
MAX_ARM_MASK_SIZE = 9

# True: ARM in pure integer mode (cpu only)
# False: ARM in integer-in-float mode (gpu or cpu)
# Both values should produce the same results.
ARMINT = False


# List of all possible scales when coding a MLP... could be moved inside
# the QuantizableModule class.
# From 0.25 256.0. Offering such high variance do saves bit!
POSSIBLE_SCALE_NN = torch.tensor([2 ** (x / 4) for x in range(-8, 32 + 1)])

# List of all possible quantization steps when coding a MLP
POSSIBLE_Q_STEP_ARM_NN = 2. ** torch.linspace(-7, 0, 8, device='cpu')
POSSIBLE_Q_STEP_SYN_NN = 2. ** torch.linspace(-16, 0, 17, device='cpu')
POSSIBLE_Q_STEP_UPS_NN = 2. ** torch.linspace(-16, 0, 17, device='cpu')

Q_PROBA_DEFAULT = 128.0

FIXED_POINT_FRACTIONAL_BITS = 6 # 8 works fine in pure int mode (ARMINT True).
                                # reduced to 6 for now for int-in-fp mode (ARMINT False)
                                # that has less headroom (23-bit mantissa, not 32)
FIXED_POINT_FRACTIONAL_MULT = 2 ** FIXED_POINT_FRACTIONAL_BITS

MAX_AC_MAX_VAL = 65535 # 2**16 for 16-bit code in bitstream header.

def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

# Trim off last bits of float. We're trying something like integerbits+8bits here.
def fix_scales():
    for idx in range(len(POSSIBLE_SCALE_NN)):
        val = POSSIBLE_SCALE_NN[idx]
        str = float_to_bin(val)
        # How many bits in the integer part?
        # We always leave the integer bits.
        # How many bits do we want in the mantissa part?
        # 1 int -> 9 mantissa
        # 2 int -> 10 mantissa
        # 8 int -> 16 mantissa
        # 16 int -> capped at 18 mantissa
        fracbits = 9
        if val < 2:
            fracbits = 9
        elif val < 4:
            fracbits = 10
        elif val < 8:
            fracbits = 11
        elif val < 16:
            fracbits = 12
        elif val < 32:
            fracbits = 13
        elif val < 64:
            fracbits = 14
        elif val < 128:
            fracbits = 15
        elif val < 256:
            fracbits = 16
        elif val < 512:
            fracbits = 17
        else:
            fracbits = 18

        frac_remove = 23-fracbits
        newstr = str[0:len(str)-frac_remove] + "0"*frac_remove
        POSSIBLE_SCALE_NN[idx] = bin_to_float(newstr)

# We rewrite the POSSIBLE_SCALE_NN values to remove trailing bits.
fix_scales()
# =============== Neural network quantization & integerization ============== #
