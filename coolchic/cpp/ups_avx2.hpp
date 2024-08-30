/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#define tostr(x) #x
#define xstr(x) tostr(x)

// kw: weights for 4 4x4 kernels.
void SYN_NAME(int KS, int32_t *kw, int h_in, int w_in, int stride_in, int32_t *in, int stride_out, int32_t *out)
{
    //printf("%s(ks=%d N_IN=%d N_OUT=%d, residue=%d, relu=%d\n", xstr(SYN_NAME), KS, N_IN, N_OUT, residue, relu);
    int const kstride = 1;
#ifdef SYN_KS
    int const ks = SYN_KS;
#else
    int const ks = KS;
#endif
    int const n_out = 4; // number of kernels, actually, always 4.

    if (ks != KS)
    {
        printf("%s: ks=%d: bad call\n", xstr(SYN_NAME), KS);
        exit(1);
    }

    int const ks2 = ks*ks;

    // we generate 8 outputs at a time, per kernel.
    // these outputs are emitted onto two output lines.
    // we advance 8 pixels in the input after that.
    int offsi_base = 0;
    int offso_base = 0;
    // the last block in x might not be full.

    int xlim_blk = ((w_in-ks+1)+7)/8; // pixels in number of horizontal blocks
    int xlast_blk_size = 8-(xlim_blk*8 - (w_in-ks+1)); // number of pixels to emit per filter in last block.
    for (int y = 0; y < h_in-ks+1; y += kstride, offsi_base += stride_in, offso_base += 2*stride_out)
    {
        int offsi_line = offsi_base;
        int offso = offso_base;
        for (int x_blk = 0; x_blk < xlim_blk; x_blk += kstride, offsi_line += kstride*8)
        {
            __m256i out_avx2[n_out];
            for (int ol = 0; ol < n_out; ol++)
            {
                out_avx2[ol] = _mm256_setzero_si256();
            }
            int offsi_block = offsi_line;
            int koffs = 0;
            for (int yy = 0; yy < ks; yy++, offsi_block += stride_in-ks)
            {
                for (int xx = 0; xx < ks; xx++, offsi_block++, koffs++)
                {
                    __m256i input = _mm256_loadu_si256((__m256i_u*)&in[offsi_block]);
                    for (int ol = 0; ol < n_out; ol++)
                    {
                        __m256i kk = _mm256_set1_epi32(kw[ol*ks2+koffs]);
                        __m256i mul = _mm256_mullo_epi32(input, kk);
                        out_avx2[ol] = _mm256_add_epi32(out_avx2[ol], mul);
                    }
                }
            }
            // merge the outputs into the destination!
            // We have:
            // ol0 A0 A1 A2 A3.. A7
            // ol1 B0 B1 B2 B3.. B7
            // ol2 C0 C1 C2 C3.. C7
            // ol3 D0 D1 D2 D3.. D7
            //
            // to go to 2 lines as:
            // A0 B0 A1 B1 A2 B2.. A7 B7
            // C0 D0 C1 D1 C2 D2.. C7 D7

            out_avx2[0] = _mm256_srai_epi32(out_avx2[0], UPS_MUL_PRECISION);
            out_avx2[1] = _mm256_srai_epi32(out_avx2[1], UPS_MUL_PRECISION);
            out_avx2[2] = _mm256_srai_epi32(out_avx2[2], UPS_MUL_PRECISION);
            out_avx2[3] = _mm256_srai_epi32(out_avx2[3], UPS_MUL_PRECISION);

            // spread horizontally and over two lines.
            //float outputs[4][8];
            __m256i outA0 = _mm256_unpacklo_epi32(out_avx2[0], out_avx2[1]);
            __m256i outA1 = _mm256_unpackhi_epi32(out_avx2[0], out_avx2[1]);
            __m256i outC0 = _mm256_unpacklo_epi32(out_avx2[2], out_avx2[3]);
            __m256i outC1 = _mm256_unpackhi_epi32(out_avx2[2], out_avx2[3]);

            // We need to remangle A0,A1 and C0,C1, switching their high and low 128-bit halves.
            __m256i outA00 = _mm256_permute2f128_si256(outA0, outA1, 0x20);
            __m256i outA01 = _mm256_permute2f128_si256(outA0, outA1, 0x31);
            __m256i outC00 = _mm256_permute2f128_si256(outC0, outC1, 0x20);
            __m256i outC01 = _mm256_permute2f128_si256(outC0, outC1, 0x31);
            if (x_blk < xlim_blk-1)
            {
                _mm256_storeu_si256((__m256i_u*)&out[offso+0], outA00);
                _mm256_storeu_si256((__m256i_u*)&out[offso+8], outA01);
                _mm256_storeu_si256((__m256i_u*)&out[offso+stride_out+0], outC00);
                _mm256_storeu_si256((__m256i_u*)&out[offso+stride_out+8], outC01);
                offso += 2*8;
            }
            else
            {
                int n_outs = 2*xlast_blk_size; // note -- doubled.
                int32_t tmp[8];
                for (int line = 0; line < 2; line++)
                {
                if (n_outs >= 8)
                {
                    _mm256_storeu_si256((__m256i_u*)&out[offso+line*stride_out+0], outA00);
                    if (n_outs >= 16)
                        _mm256_storeu_si256((__m256i_u*)&out[offso+line*stride_out+8], outA01);
                    else if (n_outs > 8)
                    {
                        // remainder over 8.
                        _mm256_storeu_si256((__m256i_u*)&tmp[0], outA01);
                        memcpy(&out[offso+line*stride_out+8], &tmp[0], (n_outs-8)*sizeof(tmp[0]));
                    }
                }
                else
                {
                    // remainder over 0.
                    _mm256_storeu_si256((__m256i_u*)&tmp[0], outA00);
                    memcpy(&out[offso+line*stride_out+0], &tmp[0], n_outs*sizeof(tmp[0]));
                }
                outA00 = outC00;
                outA01 = outC01;
                }
                offso += n_outs;
            }
        }
    }
}

#undef tostr
#undef xstr
#undef SYN_NAME
#undef SYN_KS
#undef UPS_MUL_PRECISION
