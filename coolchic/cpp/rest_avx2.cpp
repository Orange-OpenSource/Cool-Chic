/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#include <immintrin.h>

#include "TDecBinCoderCABAC.h"
#include "cc-contexts.h"
#include "common.h"
#include "rest_avx2.h"

// we have timing stuff here as well.
#include <chrono> // timing.

// FLOAT
float hsum_ps_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}

float hsum256_ps_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
           vlow  = _mm_add_ps(vlow, vhigh);     // add the low 128
    return hsum_ps_sse3(vlow);         // and inline the sse3 version, which is optimal for AVX
    // (no wasted instructions, and all of them are the 4B minimum)
}

// INT
uint32_t hsum_epi32_avx(__m128i x)
{
    __m128i hi64  = _mm_unpackhi_epi64(x, x);           // 3-operand non-destructive AVX lets us save a byte without needing a movdqa
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));    // Swap the low two elements
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);       // movd
}

// only needs AVX2
uint32_t hsum_8x32(__m256i v)
{
    __m128i sum128 = _mm_add_epi32( 
                 _mm256_castsi256_si128(v),
                 _mm256_extracti128_si256(v, 1)); // silly GCC uses a longer AXV512VL instruction if AVX512 is enabled :/
    return hsum_epi32_avx(sum128);
}

// we are applying 4 4x4 filters on a non-expanded source.
//
void custom_conv_ups_4x41_avx2(float *kw, int h_in, int w_in, float *in, float *out)
{
    int const ks = 4;
    int const stride = 1;
    int offs0 = 0;

    int32_t indexes[] = { 0, 1, 2, 3, w_in, w_in+1, w_in+2, w_in+3 };
    __m256i ind_intel = _mm256_loadu_si256((__m256i *)&indexes[0]);

    __m256 kernel_v0h0_top = _mm256_loadu_ps(kw+(0*2+0)*(ks*ks)+0);
    __m256 kernel_v0h0_bot = _mm256_loadu_ps(kw+(0*2+0)*(ks*ks)+8);
    __m256 kernel_v0h1_top = _mm256_loadu_ps(kw+(0*2+1)*(ks*ks)+0);
    __m256 kernel_v0h1_bot = _mm256_loadu_ps(kw+(0*2+1)*(ks*ks)+8);

    __m256 kernel_v1h0_top = _mm256_loadu_ps(kw+(1*2+0)*(ks*ks)+0);
    __m256 kernel_v1h0_bot = _mm256_loadu_ps(kw+(1*2+0)*(ks*ks)+8);
    __m256 kernel_v1h1_top = _mm256_loadu_ps(kw+(1*2+1)*(ks*ks)+0);
    __m256 kernel_v1h1_bot = _mm256_loadu_ps(kw+(1*2+1)*(ks*ks)+8);


    for (int y = 0; y < h_in-ks+1; y += stride, offs0 += w_in)
    {
        int offs = offs0;
        float *out_next = out+2*(w_in-ks+1);

        for (int x = 0; x < w_in-ks+1; x += stride, offs += stride)
        {
            __m256 in_0 = _mm256_i32gather_ps(&in[offs+0*w_in], ind_intel, sizeof(float)); // 8 32-bit floats
            __m256 in_1 = _mm256_i32gather_ps(&in[offs+2*w_in], ind_intel, sizeof(float)); // 8 32-bit floats

            __m256 sum_0 = kernel_v0h0_top*in_0;
            sum_0 += kernel_v0h0_bot*in_1;
            __m256 sum_1 = kernel_v0h1_top*in_0;
            sum_1 += kernel_v0h1_bot*in_1;

            *out++ = hsum256_ps_avx(sum_0);
            *out++ = hsum256_ps_avx(sum_1);

            sum_0 = kernel_v1h0_top*in_0;
            sum_0 += kernel_v1h0_bot*in_1;
            sum_1 = kernel_v1h1_top*in_0;
            sum_1 += kernel_v1h1_bot*in_1;

            *out_next++ = hsum256_ps_avx(sum_0);
            *out_next++ = hsum256_ps_avx(sum_1);
        }

        // we've done two output lines.
        out = out_next;
    }
}
