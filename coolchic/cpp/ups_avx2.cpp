/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#include <stdio.h>
#include <memory.h>
#include <immintrin.h>
#include <algorithm>
#include "common.h"

#define SYN_NAME   ups_4x4x4_fromups_avx2
#define SYN_KS     4
#define UPS_MUL_PRECISION UPS_PRECISION
#include "ups_avx2.hpp"

#define SYN_NAME   ups_4x2x2_fromups_avx2
#define SYN_KS     2
#define UPS_MUL_PRECISION UPS_PRECISION
#include "ups_avx2.hpp"

#define SYN_NAME   ups_4x4x4_fromarm_avx2
#define SYN_KS     4
#define UPS_MUL_PRECISION ARM_PRECISION
#include "ups_avx2.hpp"

#define SYN_NAME   ups_4x2x2_fromarm_avx2
#define SYN_KS     2
#define UPS_MUL_PRECISION ARM_PRECISION
#include "ups_avx2.hpp"
