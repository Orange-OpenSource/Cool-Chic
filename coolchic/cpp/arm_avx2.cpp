/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#include "TDecBinCoderCABAC.h"
#include "common.h"
#include "cc-contexts.h"
#include "cc-bac.h"
#include "arm_avx2.h"

#include <immintrin.h>

#include "common_avx2.h"

#define AVX_NAME custom_conv_11_int32_avx2_8_X_X
#define NCONTEXTS 8
#include "arm_avx2.hpp"
#undef AVX_NAME
#undef NCONTEXTS

#define AVX_NAME custom_conv_11_int32_avx2_16_X_X
#define NCONTEXTS 16
#include "arm_avx2.hpp"
#undef AVX_NAME
#undef NCONTEXTS

#define AVX_NAME custom_conv_11_int32_avx2_24_X_X
#define NCONTEXTS 24
#include "arm_avx2.hpp"
#undef AVX_NAME
#undef NCONTEXTS

#define AVX_NAME custom_conv_11_int32_avx2_32_X_X
#define NCONTEXTS 32
#include "arm_avx2.hpp"
#undef AVX_NAME
#undef NCONTEXTS

