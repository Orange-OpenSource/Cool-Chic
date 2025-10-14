/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

// Can set
// SYN_NAME   custom_conv_ks1_in7_hidden40_out3_avx2
// SYN_KS     1
// SYN_N_IN   7
// SYN_N_HIDDEN 40
// SYN_N_OUT  3

#define tostr(x) #x
#define xstr(x) tostr(x)

// Used for 7->40->3 1x1 conv, or 7->16->3 1x1 conv.
// N_IN (7)
// N_HIDDEN (16 or 40)
// N_OUT (3)
// HIDDEN always RELU
template <typename P>
void SYN_NAME(int KS,
              P *kw7_40, P *kb40, P *kw40_3, P *kb3,
              int h_in, int w_in, int stride_in, int plane_stride_in, int N_IN, int N_HIDDEN, P *in, int N_OUT, P *out)
{
    //printf("%s(ks=%d N_IN=%d N_HIDDEN=%d N_OUT=%d\n", xstr(SYN_NAME), KS, N_IN, N_HIDDEN, N_OUT);

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
#ifdef SYN_N_HIDDEN
    int const n_hidden = SYN_N_HIDDEN;
#else
    int const n_hidden = N_HIDDEN;
#endif
#ifdef SYN_N_OUT
    int const n_out = SYN_N_OUT;
#else
    int const n_out = N_OUT;
#endif
    int const src_pad = (stride_in-w_in)/2;

    int const n_hidden8 = n_hidden/8;

    P *src = in;
    P *dst = out;

    if (KS != ks)
    {
        printf("%s: bad call: ks=%d must be 1\n", xstr(SYN_NAME), KS);
        exit(1);
    }
    if (n_in != 7 && n_in != 8)
    {
        printf("%s: bad call: n_in %d can only be 7 or 8\n", xstr(SYN_NAME), n_in);
        exit(1);
    }
    if (n_hidden%8 != 0)
    {
        printf("%s: bad call: n_hidden %d not multiple of 8\n", xstr(SYN_NAME), n_hidden);
        exit(1);
    }

    // gather-indicies
    int32_t pixel_indicies[8] = { 0*plane_stride_in, 1*plane_stride_in, 2*plane_stride_in, 3*plane_stride_in,
                                  4*plane_stride_in, 5*plane_stride_in, 6*plane_stride_in, n_in < 8 ? 0 : 7*plane_stride_in};
    __m256i ind_intel_0 = _mm256_loadu_si256((__m256i *)&pixel_indicies[0*8]);

    const __m256i rotate_right = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
    const __m256i z = _mm256_setzero_si256();

    for (int y = 0; y < h_in; y++, src += src_pad+src_pad, dst += src_pad+src_pad) // eol of this, and bol of next.
    for (int x = 0; x < w_in; x++, src++, dst++)
    {
        __m256i input_avx2_src = _mm256_i32gather_epi32((int32_t *)&src[0], ind_intel_0, sizeof(int32_t));
        __m256i hidden_avx2[n_hidden/8];

        //for (int hl = 0; hl < n_hidden_layers; hl++)
        {
            P *kw = kw7_40;

            for (int h8 = 0; h8 < n_hidden/8; h8++)
            {
                hidden_avx2[h8] = _mm256_loadu_si256((const __m256i*)(kb40+h8*8));
            }

            for (int il = 0; il < n_in; il++, kw += n_hidden)
            {
                __m256i input_avx2_pixel = _mm256_broadcastd_epi32(_mm256_castsi256_si128(input_avx2_src));
                input_avx2_src = _mm256_permutevar8x32_epi32(input_avx2_src, rotate_right);

                for (int h8 = 0; h8 < n_hidden/8; h8++)
                {
                    if constexpr(std::is_same<P, int32_t>::value)
                    {
                        __m256i mul_avx2 = _mm256_loadu_si256((const __m256i *)(kw+h8*8));
                        mul_avx2 = _mm256_mullo_epi32(input_avx2_pixel, mul_avx2);
                        hidden_avx2[h8] = _mm256_add_epi32(hidden_avx2[h8], mul_avx2);
                    }
                    else
                    {
                        __m256 mul_avx2 = (__m256)_mm256_loadu_si256((const __m256i *)(kw+h8*8));
                        mul_avx2 = _mm256_mul_ps((__m256)input_avx2_pixel, mul_avx2);
                        hidden_avx2[h8] = (__m256i)_mm256_add_ps((__m256)hidden_avx2[h8], mul_avx2);
                    }
                }
            }

            // always relu.
            for (int h8 = 0; h8 < n_hidden/8; h8++)
            {
                if constexpr(std::is_same<P, int32_t>::value)
                {
                    hidden_avx2[h8] = _mm256_blendv_epi8(z, hidden_avx2[h8], _mm256_cmpgt_epi32(hidden_avx2[h8], z));
                    hidden_avx2[h8] = _mm256_srai_epi32(hidden_avx2[h8], SYN_MUL_PRECISION);
                }
                else
                {
                    hidden_avx2[h8] = _mm256_blendv_epi8(z, hidden_avx2[h8], (__m256i) _mm256_cmp_ps((__m256)hidden_avx2[h8], (__m256)z, _CMP_GT_OS));
                }
            }
        }

        // FINAL 40 -> 3
        P *kw = kw40_3;
        P *kb = kb3;
        for (int ol = 0; ol < n_out; ol++)
        {

            __m256i out_avx2 = _mm256_setzero_si256();
            for (int h8 = 0; h8 < n_hidden8; h8++, kw += 8)
            {
                if constexpr(std::is_same<P, int32_t>::value)
                {
                    __m256i mul_avx2 = _mm256_loadu_si256((const __m256i *)kw);
                    mul_avx2 = _mm256_mullo_epi32(mul_avx2, hidden_avx2[h8]);
                    out_avx2 = _mm256_add_epi32(out_avx2, mul_avx2);
                }
                else
                {
                    __m256 mul_avx2 = _mm256_loadu_ps(kw);
                    mul_avx2 = _mm256_mul_ps(mul_avx2, (__m256)hidden_avx2[h8]);
                    out_avx2 = (__m256i)_mm256_add_ps((__m256)out_avx2, mul_avx2);
                }
            }

            // horizontal sum
            P sum;
            if constexpr(std::is_same<P, int32_t>::value)
            {
                sum = kb[ol] + hsum_8x32(out_avx2);
                if (sum < 0)
                    sum = -(-sum >> SYN_MUL_PRECISION);
                else
                    sum >>= SYN_MUL_PRECISION;
            }
            else
            {
                sum = kb[ol] + horizontal_add((__m256)out_avx2);
            }
            dst[ol*plane_stride_in] = sum;
        }
    } // x, y
}

// instantiate int32_t and float
template
void SYN_NAME<int32_t>(int KS, int32_t *kw7_40, int32_t *kb40, int32_t *kw40_3, int32_t *kb3, int h_in, int w_in, int stride_in, int plane_stride_in, int N_IN, int N_HIDDEN, int32_t *in, int N_OUT, int32_t *out);
template
void SYN_NAME<float>(int KS, float *kw7_40, float *kb40, float *kw40_3, float *kb3, int h_in, int w_in, int stride_in, int plane_stride_in, int N_IN, int N_HIDDEN, float *in, int N_OUT, float *out);

#undef SYN_NAME
#undef SYN_KS
#undef SYN_N_IN
#undef SYN_N_HIDDEN
#undef SYN_N_OUT
