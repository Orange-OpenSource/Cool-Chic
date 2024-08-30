/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


// TO BE INCLUDED WITH AVX_NAME AND NCONTEXTS DEFINED.


void AVX_NAME(weights_biases *kwtX_n_n, weights_biases *kbX_n, // kwtX_n_n[n_hidden_layers] -- kernel weights, transposed.
                                      weights_biases *kwOUT_n_2, weights_biases *kbOUT_2, // _n_2, weights not transposed.
                                      int32_t *context_indicies, int32_t n_contexts_param, int32_t n_hidden_layers,
                                      int32_t *src,
                                      int src_h, int src_w, int src_pad,
                                      BACContext &bac_context
                                      )
{
    int const n_in = NCONTEXTS; // n_contexts;
    int const n_inout8 = n_in/8;
    int const n_final_out = 2;

    if (n_contexts_param != n_in)
    {
        printf("bad avx2_X_X_X call: context_param %d wanted %d\n", n_contexts_param, n_in);
        exit(1);
    }

    const __m256i m64_avx2 = _mm256_set1_epi32(ARM_SCALE);
    const __m256i rotate_right = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
    const __m256i z = _mm256_setzero_si256();
    __m256i mul_avx2;

    // gather-indicies
    __m256i ind_intel_0 = _mm256_loadu_si256((__m256i *)&context_indicies[0*8]);
#if NCONTEXTS >= 16
    __m256i ind_intel_1 = _mm256_loadu_si256((__m256i *)&context_indicies[1*8]);
#endif
#if NCONTEXTS >= 24
    __m256i ind_intel_2 = _mm256_loadu_si256((__m256i *)&context_indicies[2*8]);
#endif
#if NCONTEXTS >= 32
    __m256i ind_intel_3 = _mm256_loadu_si256((__m256i *)&context_indicies[3*8]);
#endif

    for (int y = 0; y < src_h; y++, src += src_pad+src_pad) // eol of this, and bol of next.
    for (int x = 0; x < src_w; x++, src++)
    {
        if (!bac_coded(bac_context, y, x))
        {
            src[0] = 0;
            continue;
        }
        int use_left = 1;
        if (bac_flat(bac_context, y, x, use_left))
        {
            if (use_left)
            {
                src[0] = src[-1];
                continue;
            }
            else
            {
                src[0] = src[-(src_w+src_pad+src_pad)];
                continue;
            }
        }

        __m256i input_avx2_src0_0;
        __m256i out_avx2_src0_0;
#if NCONTEXTS >= 16
        __m256i input_avx2_src0_1;
        __m256i out_avx2_src0_1;
#endif
#if NCONTEXTS >= 24
        __m256i input_avx2_src0_2;
        __m256i out_avx2_src0_2;
#endif
#if NCONTEXTS >= 32
        __m256i input_avx2_src0_3;
        __m256i out_avx2_src0_3;
#endif

        input_avx2_src0_0 = _mm256_i32gather_epi32(&src[0], ind_intel_0, sizeof(int32_t));
#if NCONTEXTS >= 16
        input_avx2_src0_1 = _mm256_i32gather_epi32(&src[0], ind_intel_1, sizeof(int32_t));
#endif
#if NCONTEXTS >= 24
        input_avx2_src0_2 = _mm256_i32gather_epi32(&src[0], ind_intel_2, sizeof(int32_t));
#endif
#if NCONTEXTS >= 32
        input_avx2_src0_3 = _mm256_i32gather_epi32(&src[0], ind_intel_3, sizeof(int32_t));
#endif

        for (int hl = 0; hl < n_hidden_layers; hl++)
        {
            int32_t *k = kwtX_n_n[hl].data;
            out_avx2_src0_0 = _mm256_load_si256((const __m256i*)(kbX_n[hl].data+0*8));
#if NCONTEXTS >= 16
            out_avx2_src0_1 = _mm256_load_si256((const __m256i*)(kbX_n[hl].data+1*8));
#endif
#if NCONTEXTS >= 24
            out_avx2_src0_2 = _mm256_load_si256((const __m256i*)(kbX_n[hl].data+2*8));
#endif
#if NCONTEXTS >= 32
            out_avx2_src0_3 = _mm256_load_si256((const __m256i*)(kbX_n[hl].data+3*8));
#endif

            __m256i in_avx2;

            if (1) // residual=1
            {
                mul_avx2 = _mm256_mullo_epi32(input_avx2_src0_0, m64_avx2);
                out_avx2_src0_0 = _mm256_add_epi32(out_avx2_src0_0, mul_avx2);
#if NCONTEXTS >= 16
                mul_avx2 = _mm256_mullo_epi32(input_avx2_src0_1, m64_avx2);
                out_avx2_src0_1 = _mm256_add_epi32(out_avx2_src0_1, mul_avx2);
#endif
#if NCONTEXTS >= 24
                mul_avx2 = _mm256_mullo_epi32(input_avx2_src0_2, m64_avx2);
                out_avx2_src0_2 = _mm256_add_epi32(out_avx2_src0_2, mul_avx2);
#endif
#if NCONTEXTS >= 32
                mul_avx2 = _mm256_mullo_epi32(input_avx2_src0_3, m64_avx2);
                out_avx2_src0_3 = _mm256_add_epi32(out_avx2_src0_3, mul_avx2);
#endif
            }

            // do in lots of 8.
            for (int il8 = 0; il8 < n_inout8; il8++)
            {
                for (int sub = 0; sub < 8; sub++, k += n_in)
                {
                    /* broadcast to 16 outputs arrving in output registers */
                    in_avx2 = _mm256_broadcastd_epi32(_mm256_castsi256_si128(input_avx2_src0_0));
                    input_avx2_src0_0 = _mm256_permutevar8x32_epi32( input_avx2_src0_0, rotate_right );

                    mul_avx2 = _mm256_load_si256((const __m256i *)(k));
                    mul_avx2 = _mm256_mullo_epi32(in_avx2, mul_avx2);
                    out_avx2_src0_0 = _mm256_add_epi32(out_avx2_src0_0, mul_avx2);
#if NCONTEXTS >= 16
                    mul_avx2 = _mm256_load_si256((const __m256i *)(k+8));
                    mul_avx2 = _mm256_mullo_epi32(in_avx2, mul_avx2);
                    out_avx2_src0_1 = _mm256_add_epi32(out_avx2_src0_1, mul_avx2);
#endif
#if NCONTEXTS >= 24
                    mul_avx2 = _mm256_load_si256((const __m256i *)(k+16));
                    mul_avx2 = _mm256_mullo_epi32(in_avx2, mul_avx2);
                    out_avx2_src0_2 = _mm256_add_epi32(out_avx2_src0_2, mul_avx2);
#endif
#if NCONTEXTS >= 32
                    mul_avx2 = _mm256_load_si256((const __m256i *)(k+24));
                    mul_avx2 = _mm256_mullo_epi32(in_avx2, mul_avx2);
                    out_avx2_src0_3 = _mm256_add_epi32(out_avx2_src0_3, mul_avx2);
#endif
                }
                // next input.
#if NCONTEXTS == 8
                // nothing.
#elif NCONTEXTS == 16
                input_avx2_src0_0 = input_avx2_src0_1;
#elif NCONTEXTS == 24
                if (il8 == 0)
                    input_avx2_src0_0 = input_avx2_src0_1;
                else
                    input_avx2_src0_0 = input_avx2_src0_2;
#elif NCONTEXTS == 32
                if (il8 == 0)
                    input_avx2_src0_0 = input_avx2_src0_1;
                else if (il8 == 1)
                    input_avx2_src0_0 = input_avx2_src0_2;
                else
                    input_avx2_src0_0 = input_avx2_src0_3;
#else
                what NCONTEXTS
#endif
            }

            out_avx2_src0_0 = _mm256_blendv_epi8(z, out_avx2_src0_0, _mm256_cmpgt_epi32(out_avx2_src0_0, z));
#if NCONTEXTS >= 16
            out_avx2_src0_1 = _mm256_blendv_epi8(z, out_avx2_src0_1, _mm256_cmpgt_epi32(out_avx2_src0_1, z));
#endif
#if NCONTEXTS >= 24
            out_avx2_src0_2 = _mm256_blendv_epi8(z, out_avx2_src0_2, _mm256_cmpgt_epi32(out_avx2_src0_2, z));
#endif
#if NCONTEXTS >= 32
            out_avx2_src0_3 = _mm256_blendv_epi8(z, out_avx2_src0_3, _mm256_cmpgt_epi32(out_avx2_src0_3, z));
#endif

            in_avx2 = _mm256_broadcastd_epi32(_mm_set1_epi32(ARM_SCALE/2));

            out_avx2_src0_0 = _mm256_add_epi32(out_avx2_src0_0, in_avx2);
            input_avx2_src0_0 = _mm256_srai_epi32(out_avx2_src0_0, ARM_PRECISION); // /ARM_SCALE -- src value is positive.
#if NCONTEXTS >= 16
            out_avx2_src0_1 = _mm256_add_epi32(out_avx2_src0_1, in_avx2);
            input_avx2_src0_1 = _mm256_srai_epi32(out_avx2_src0_1, ARM_PRECISION); // /ARM_SCALE -- src value is positive.
#endif
#if NCONTEXTS >= 24
            out_avx2_src0_2 = _mm256_add_epi32(out_avx2_src0_2, in_avx2);
            input_avx2_src0_2 = _mm256_srai_epi32(out_avx2_src0_2, ARM_PRECISION); // /ARM_SCALE -- src value is positive.
#endif
#if NCONTEXTS >= 32
            out_avx2_src0_3 = _mm256_add_epi32(out_avx2_src0_3, in_avx2);
            input_avx2_src0_3 = _mm256_srai_epi32(out_avx2_src0_3, ARM_PRECISION); // /ARM_SCALE -- src value is positive.
#endif
        }

        // FINAL N -> 2

        // not like above -- there are only two outputs here.
        // we want full registers for the multiplies, and wear the overhead
        // of a horizontal sum for each of the two outputs.
        int out[2];
        int32_t *k = kwOUT_n_2->data;
        for (int ol = 0; ol < n_final_out; ol++)
        {
            mul_avx2 = _mm256_load_si256((const __m256i *)k); k += 8;
            out_avx2_src0_0 = _mm256_mullo_epi32(mul_avx2, input_avx2_src0_0);
#if NCONTEXTS >= 16
            mul_avx2 = _mm256_load_si256((const __m256i *)k); k += 8;
            mul_avx2 = _mm256_mullo_epi32(mul_avx2, input_avx2_src0_1);
            out_avx2_src0_0 = _mm256_add_epi32(out_avx2_src0_0, mul_avx2);
#endif
#if NCONTEXTS >= 24
            mul_avx2 = _mm256_load_si256((const __m256i *)k); k += 8;
            mul_avx2 = _mm256_mullo_epi32(mul_avx2, input_avx2_src0_2);
            out_avx2_src0_0 = _mm256_add_epi32(out_avx2_src0_0, mul_avx2);
#endif
#if NCONTEXTS >= 32
            mul_avx2 = _mm256_load_si256((const __m256i *)k); k += 8;
            mul_avx2 = _mm256_mullo_epi32(mul_avx2, input_avx2_src0_3);
            out_avx2_src0_0 = _mm256_add_epi32(out_avx2_src0_0, mul_avx2);
#endif

            // horizontal sum
            int32_t sum = kbOUT_2->data[ol] + hsum_8x32(out_avx2_src0_0);
            if (sum < 0)
                sum = -((-sum+ARM_SCALE/2) >> ARM_PRECISION);
            else
                sum = (sum+ARM_SCALE/2) >> ARM_PRECISION;

            // store.
            out[ol] = sum;
        }

        // bac it.
        int xx = decode_latent_layer_bac_single(
                        bac_context,
                        out[0], out[1]
                    );
        src[0] = xx<<ARM_PRECISION;
    } // x, y
}
