# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


"""Generate once and for all a table containing all the possible values
for mu and scales.
"""

import numpy as np
import torch

from coolchic.bitstream.component.constants import (
    AC_MAX_VAL,
    FRAC_ACCURACY_MU_SCALE,
    MU_MAX,
    MU_MIN,
    PATH_MU_SCALE_TABLE,
)
from coolchic.component.core.arm import LOG_SCALE_MAX, LOG_SCALE_MIN

if __name__ == "__main__":
    # mu is in [MU_MIN, MU_MAX] with MU_MAX included
    mu = torch.arange(MU_MIN, MU_MAX + FRAC_ACCURACY_MU_SCALE, FRAC_ACCURACY_MU_SCALE)
    log_scale = torch.arange(
        LOG_SCALE_MIN, LOG_SCALE_MAX + FRAC_ACCURACY_MU_SCALE, FRAC_ACCURACY_MU_SCALE
    )
    latent = torch.arange(-AC_MAX_VAL, AC_MAX_VAL, 1)
    scale = log_scale.exp()

    data_to_save = torch.cat([mu, scale], dim=0).numpy().astype(np.float32)
    np.save(PATH_MU_SCALE_TABLE, data_to_save)
