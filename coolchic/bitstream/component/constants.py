import os

import torch

from coolchic.component.core.arm import LOG_SCALE_MAX, LOG_SCALE_MIN

FIXED_POINT_DTYPE = torch.int64

# Latent values will be always be in [-AC_MAX_VAL, AC_MAX_VAL - 1]
# when writing the bitstream
AC_MAX_VAL = 64

# Where the probability table is
PATH_MU_SCALE_TABLE = f"{os.path.dirname(os.path.realpath(__file__))}/mu_scale.npy"


WEIGHT_SHIFT = 16  # scale is 2 ** -WEIGHT_SHIFT
WEIGHT_SHIFT_MULT = 2**WEIGHT_SHIFT
WEIGHT_INV_SHIFT_MULT = 2**-WEIGHT_SHIFT

BIAS_SHIFT = WEIGHT_SHIFT  # 2 * WEIGHT_SHIFT  # scale is 2 ** -BIAS_SHIFT
BIAS_SHIFT_MULT = 2**BIAS_SHIFT

N_FRAC_BIT_MU_SCALE = 8
FRAC_ACCURACY_MU_SCALE = 2**-N_FRAC_BIT_MU_SCALE

# Mu will be in [-AC_MAX_VAL, AC_MAX_VAL - FRAC_ACCURACY_MU]
MU_MIN = -AC_MAX_VAL
MU_MAX = AC_MAX_VAL - FRAC_ACCURACY_MU_SCALE
# +1 because MU_MAX is included
N_POSSIBLE_MU = int((MU_MAX - MU_MIN) // FRAC_ACCURACY_MU_SCALE + 1)
MU_MIN_FIXED_POINT = MU_MIN << N_FRAC_BIT_MU_SCALE


# +1 because LOG_SCALE_MAX is included
N_POSSIBLE_SCALE = int((LOG_SCALE_MAX - LOG_SCALE_MIN) // FRAC_ACCURACY_MU_SCALE + 1)
LOG_SCALE_MIN_FIXED_POINT = LOG_SCALE_MIN << N_FRAC_BIT_MU_SCALE

N_FRAC_BIT_INTER_FT_CTX = 8  # N_FRAC_BIT_INTER_FT_CTX + min qstep arm <= WEIGHT_SHIFT
