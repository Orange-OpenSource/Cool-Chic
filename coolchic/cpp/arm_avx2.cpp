/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#include "TDecBinCoderCABAC.h"
#include "cc-contexts.h"
#include "common.h"
#include "arm_avx2.h"

#include <immintrin.h>

// INT
static uint32_t hsum_epi32_avx(__m128i x)
{
    __m128i hi64  = _mm_unpackhi_epi64(x, x);           // 3-operand non-destructive AVX lets us save a byte without needing a movdqa
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));    // Swap the low two elements
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);       // movd
}

// only needs AVX2
static uint32_t hsum_8x32(__m256i v)
{
    __m128i sum128 = _mm_add_epi32( 
                 _mm256_castsi256_si128(v),
                 _mm256_extracti128_si256(v, 1)); // silly GCC uses a longer AXV512VL instruction if AVX512 is enabled :/
    return hsum_epi32_avx(sum128);
}

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

