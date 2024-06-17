/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#include <stdio.h>
#include <immintrin.h>

// Can set
// SYN_NAME
// SYN_KS
// SYN_N_IN
// SYN_N_OUT

#define tostr(x) #x
#define xstr(x) tostr(x)

// only suitable for width%8 == 0

// POSSIBLE IN-PLACE -- output overwrites input, to minimise cpu cache stress with multi-layer approach.
// ALWAYS IN-PLACE for SYN_KS == 1
// stride_in and plane_stride_in are assumed the same for out.
void SYN_NAME(int KS, float *kw, float *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, float *in, int N_OUT, float *out, int residue, int relu)
{
    //printf("%s(ks=%d N_IN=%d N_OUT=%d, residue=%d, relu=%d\n", xstr(SYN_NAME), KS, N_IN, N_OUT, residue, relu);
    int const kstride = 1;
#ifdef SYN_KS
    int const ks = SYN_KS;
#else
    int const ks = KS;
#endif
#ifdef SYN_N_IN
    int const n_in = SYN_N_IN;
#else
    int const n_in = N_IN;
#endif
#ifdef SYN_N_OUT
    int const n_out = SYN_N_OUT;
#else
    int const n_out = N_OUT;
#endif

    float *in_layer[std::max(n_in, n_out)];
    in_layer[0] = in;
    for (int i = 1; i < std::max(n_in, n_out); i++)
        in_layer[i] = in_layer[i-1]+plane_stride_in;
#if SYN_KS == 1
// always in-place for ks==1
#define out_layer in_layer
    if (out != NULL && out != in)
    {
        printf("%s: ks=%d n_in=%d n_out=%d: bad call: should be in-place, but out supplied\n", xstr(SYN_NAME), KS, N_IN, N_OUT);
        exit(1);
    }
#else
    //not in-place
    float *out_layer[N_OUT];
    out_layer[0] = out;
    for (int i = 1; i < N_OUT; i++)
        out_layer[i] = out_layer[i-1]+plane_stride_in;
#endif

    const __m256 z = _mm256_setzero_ps();

    // 8 at a time.
    int xlim_blk = ((w_in-ks+1)+7)/8; // pixels in this number of horizontal blocks, last may not be full.
    int xlast_blk_size = 8-(xlim_blk*8 - (w_in-ks+1)); // number of pixels to emit per filter in last block.
    if (ks != KS || n_in != N_IN || n_out != N_OUT)
    {
        printf("%s: ks=%d n_in=%d n_out=%d: bad call\n", xstr(SYN_NAME), KS, N_IN, N_OUT);
        exit(1);
    }

    int const ks2 = ks*ks;

    int offsi_base = 0;
    for (int y = 0; y < h_in-ks+1; y += kstride, offsi_base += stride_in)
    {
        int offsi_line = offsi_base;
        int offso = offsi_base;
        for (int x_blk = 0; x_blk < xlim_blk; x_blk += kstride, offsi_line += kstride*8)
        {
            __m256 out_avx2[n_out];
            // initialize with bias.
            for (int ol = 0; ol < n_out; ol++)
            {
                out_avx2[ol] = _mm256_broadcast_ss(&kb[ol]);
            }
            if (residue)
            {
                for (int ol = 0; ol < n_out; ol++)
                {
                    // __m256 in = _mm256_loadu_ps(&out_layer[ol][offso]); // !!! we are assuming the output has been primed with (unpadded) input!
                    __m256 in = _mm256_loadu_ps(&in_layer[ol][offso+residue_origin_offset]);
                    out_avx2[ol] = _mm256_add_ps(out_avx2[ol], in);
                }
            }
            for (int il = 0; il < n_in; il++)
            {
                int offsi_block = offsi_line;
                int koffs = 0;
                for (int yy = 0; yy < ks; yy++, offsi_block += stride_in-ks)
                {
                    for (int xx = 0; xx < ks; xx++, offsi_block++, koffs++)
                    {
                        __m256 in = _mm256_loadu_ps(&in_layer[il][offsi_block]);
                        for (int ol = 0; ol < n_out; ol++)
                        {
                            __m256 kk = _mm256_broadcast_ss(&kw[ol*n_in*ks2+il*ks2+koffs]);
                            __m256 mul = _mm256_mul_ps(in, kk);
                            out_avx2[ol] = _mm256_add_ps(out_avx2[ol], mul);
                        }
                    }
                }
            }
            if (relu)
            {
                for (int ol = 0; ol < n_out; ol++)
                {
                    out_avx2[ol] = _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(z), _mm256_castps_si256(out_avx2[ol]), _mm256_castps_si256(_mm256_cmp_ps(out_avx2[ol], z, _CMP_GT_OQ))));
                }
            }
            if (x_blk < xlim_blk-1)
            {
                for (int ol = 0; ol < n_out; ol++)
                {
                    _mm256_storeu_ps(&out_layer[ol][offso], out_avx2[ol]);
                }
                offso += 8;
            }
            else
            {
                int n_outs = xlast_blk_size;
                // partial last line.
                for (int ol = 0; ol < n_out; ol++)
                {
                    float store[8];
                    _mm256_storeu_ps(&store[0], out_avx2[ol]);
                    memcpy(&out_layer[ol][offso], &store[0], n_outs*sizeof(store[0]));
                }
                offso += n_outs;
            }
        }
    }
}

#undef tostr
#undef xstr
#undef out_layer
#undef SYN_NAME
#undef SYN_KS
#undef SYN_N_IN
#undef SYN_N_OUT
