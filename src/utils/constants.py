# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
import struct

LIST_POSSIBLE_DEVICES = ['cpu', 'cuda:0', 'mps:0']

# True: ARM in pure integer mode (cpu only)
# False: ARM in integer-in-float mode (gpu or cpu)
# Both values should produce the same results.
ARMINT = False

# Avoid numerical instability when measuring the rate of the NN parameters
MIN_SCALE_NN_WEIGHTS_BIAS = 1.0/1024.0

# List of all possible scales when coding a MLP
POSSIBLE_SCALE_NN = 2 ** torch.linspace(
    MIN_SCALE_NN_WEIGHTS_BIAS, 16, steps=2 ** 16 - 1, device='cpu'
)
# List of all possible quantization steps when coding a MLP
POSSIBLE_Q_STEP_ARM_NN = 2. ** torch.linspace(-7, 0, 8, device='cpu')
POSSIBLE_Q_STEP_SYN_NN = 2. ** torch.linspace(-16, 0, 17, device='cpu')
POSSIBLE_Q_STEP_UPS_NN = 2. ** torch.linspace(-16, 0, 17, device='cpu')

Q_PROBA_DEFAULT = 128.0

FIXED_POINT_FRACTIONAL_BITS = 6 # 8 works fine in pure int mode (ARMINT True).
                                # reduced to 6 for now for int-in-fp mode (ARMINT False)
                                # that has less headroom (23-bit mantissa, not 32)
FIXED_POINT_FRACTIONAL_MULT = 2**FIXED_POINT_FRACTIONAL_BITS

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
