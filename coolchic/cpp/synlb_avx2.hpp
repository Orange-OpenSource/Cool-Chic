/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


// Can set
// SYN_NAME
// SYN_KS
// SYN_N_IN
// SYN_N_OUT

#define tostr(x) #x
#define xstr(x) tostr(x)

// ALWAYS IN-PLACE for SYN_KS == 1
// stride_in and plane_stride_in are assumed the same for out.
// dedicated to 3x3 kernels that use a line-buffer for temporary storage, allowing in-place convolution.
template <typename P>
void SYN_NAME(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, P *line_buffer, int residue, int relu)
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
#ifdef SYN_RES
#if SYN_RES == 0
    if (residue)
    {
        printf("%s( residue supplied but not expected)\n", xstr(SYN_NAME));
        exit(1);
    }
#else
    if (!residue)
    {
        printf("%s( residue not supplied but expected)\n", xstr(SYN_NAME));
        exit(1);
    }
#endif
#endif

    P *in_layer[n_in];
    in_layer[0] = in;
    for (int i = 1; i < n_in; i++)
        in_layer[i] = in_layer[i-1]+plane_stride_in;

    P *out_layer[n_out];
    out_layer[0] = out;
    for (int i = 1; i < n_out; i++)
        out_layer[i] = out_layer[i-1]+plane_stride_in;

    // in-place, must have line buffer.
    P *lb[2]; // two line buffer pointers.
    int h_out = h_in-ks+1;
    int w_out = w_in-ks+1;
    if (ks != 3)
    {
        printf("%s: bad call: in-place lb must have ks=3\n", xstr(SYN_NAME));
        exit(1);
    }
    // must have a line buffer.
    if (line_buffer == NULL)
    {
        printf("%s: bad call, no line buffer supplied\n", xstr(SYN_NAME));
        exit(1);
    }
    lb[0] = line_buffer;
    lb[1] = line_buffer+w_out*n_out;

    const __m256i z = _mm256_setzero_si256();

    // 8 at a time.
    int xlim_blk = ((w_in-ks+1)+(8-1))/8; // pixels in this number of horizontal blocks, last may not be full.
    int xlast_blk_size = 8-(xlim_blk*8 - (w_in-ks+1)); // number of pixels to emit per filter in last block.
    if (ks != KS || n_in != N_IN || n_out != N_OUT)
    {
        printf("%s: ks=%d n_in=%d n_out=%d: bad call\n", xstr(SYN_NAME), KS, N_IN, N_OUT);
        exit(1);
    }

    int const ks2 = ks*ks;

    int offsi_base = 0;

#ifdef SYN_ATATIME
    // manage large numbers of outputs.
    int const atatime = SYN_ATATIME;
#else
    // assuming small numbers of outputs.
    int const atatime = n_out;
#endif

    for (int y = 0; y < h_in-ks+1; y += kstride, offsi_base += stride_in)
    {
        int offsi_line = offsi_base;
        int offso = offsi_base;
        for (int x_blk = 0; x_blk < xlim_blk; x_blk += kstride, offsi_line += kstride*8)
        {
            for (int olbase = 0; olbase < n_out; olbase += atatime)
            {
                __m256i out_avx2[atatime];
                // initialize with bias.
                for (int ol = 0; ol < atatime; ol++)
                {
                    if constexpr(std::is_same<P, int32_t>::value)
                        out_avx2[ol] = _mm256_set1_epi32(kb[olbase+ol]);
                    else
                        out_avx2[ol] = (__m256i) _mm256_set1_ps(kb[olbase+ol]);
                }
                if (residue)
                {
                    for (int ol = 0; ol < atatime; ol++)
                    {
                        __m256i in = _mm256_loadu_si256((const __m256i_u*)&in_layer[olbase+ol][offso+residue_origin_offset]);
                        if constexpr(std::is_same<P, int32_t>::value)
                        {
                            in = _mm256_slli_epi32(in, SYN_MUL_PRECISION);
                            out_avx2[ol] = _mm256_add_epi32(out_avx2[ol], in);
                        }
                        else
                        {
                            out_avx2[ol] = (__m256i)_mm256_add_ps((__m256)out_avx2[ol], (__m256)in);
                        }
                    }
                }
                for (int il = 0; il < n_in; il++)
                {
                    int offsi_block = offsi_line;
                    int koffs = olbase*n_in*ks2+il*ks2;
                    for (int yy = 0; yy < ks; yy++, offsi_block += stride_in-ks)
                    {
                        for (int xx = 0; xx < ks; xx++, offsi_block++, koffs++)
                        {
                            __m256i in = _mm256_loadu_si256((const __m256i_u*)&in_layer[il][offsi_block]);
                            int kkoffs = koffs;
                            for (int ol = 0; ol < atatime; ol++, kkoffs += n_in*ks2)
                            {
                                if constexpr(std::is_same<P, int32_t>::value)
                                {
                                    __m256i kk = _mm256_set1_epi32(kw[kkoffs]);
                                    __m256i mul = _mm256_mullo_epi32(in, kk);
                                    out_avx2[ol] = _mm256_add_epi32(out_avx2[ol], mul);
                                }
                                else
                                {
                                    __m256 kk = _mm256_set1_ps(kw[kkoffs]);
                                    __m256 mul = _mm256_mul_ps((__m256)in, kk);
                                    out_avx2[ol] = (__m256i)_mm256_add_ps((__m256)out_avx2[ol], mul);
                                }
                            }
                        }
                    }
                }
                int n_outs = (x_blk < xlim_blk-1) ? 8 : xlast_blk_size;
                int32_t store[8];
                if constexpr(std::is_same<P, int32_t>::value)
                {
                    if (relu)
                    {
                        // -ves to zero, >> for rest.
                        for (int ol = 0; ol < atatime; ol++)
                        {
                            out_avx2[ol] = _mm256_blendv_epi8(z, out_avx2[ol], _mm256_cmpgt_epi32(out_avx2[ol], z));
                            out_avx2[ol] = _mm256_srai_epi32(out_avx2[ol], SYN_MUL_PRECISION);
                            if (n_outs == 8)
                            {
                                _mm256_storeu_si256((__m256i_u*)&lb[y%2][(olbase+ol)*w_out+x_blk*8], out_avx2[ol]);
                            }
                            else
                            {
                                _mm256_storeu_si256((__m256i_u*)&store[0], out_avx2[ol]);
                                memcpy(&lb[y%2][(olbase+ol)*w_out+x_blk*8], &store[0], n_outs*sizeof(store[0]));
                            }
                        }
                    }
                    else
                    {
                        // need different treatment for -ve and +ve.
                        for (int ol = 0; ol < atatime; ol++)
                        {
                            __m256i sr  = _mm256_srai_epi32(out_avx2[ol], SYN_MUL_PRECISION);
                            __m256i negsrneg = _mm256_sub_epi32(z, _mm256_srai_epi32(_mm256_sub_epi32(z, out_avx2[ol]), SYN_MUL_PRECISION));
                            out_avx2[ol] = _mm256_blendv_epi8(negsrneg, sr, _mm256_cmpgt_epi32(out_avx2[ol], z));
                            if (n_outs == 8)
                            {
                                _mm256_storeu_si256((__m256i_u*)&lb[y%2][(olbase+ol)*w_out+x_blk*8], out_avx2[ol]);
                            }
                            else
                            {
                                _mm256_storeu_si256((__m256i_u*)&store[0], out_avx2[ol]);
                                memcpy(&lb[y%2][(olbase+ol)*w_out+x_blk*8], &store[0], n_outs*sizeof(store[0]));
                            }
                        }
                    }
                }
                else
                {
                    for (int ol = 0; ol < atatime; ol++)
                    {
                        if (relu)
                            out_avx2[ol] = _mm256_blendv_epi8(z, out_avx2[ol], (__m256i) _mm256_cmp_ps((__m256)out_avx2[ol], (__m256)z, _CMP_GT_OS));
                        if (n_outs == 8)
                        {
                            _mm256_storeu_si256((__m256i_u*)&lb[y%2][(olbase+ol)*w_out+x_blk*8], out_avx2[ol]);
                        }
                        else
                        {
                            _mm256_storeu_si256((__m256i_u*)&store[0], out_avx2[ol]);
                            memcpy(&lb[y%2][(olbase+ol)*w_out+x_blk*8], &store[0], n_outs*sizeof(store[0]));
                        }
                    }
                }
            } // olbase
            offso += 8;
        } // x_blk
        // we are at the end of this line, flush previous line buffer, if any.
        if (y >= 1)
        {
            for (int ol = 0; ol < n_out; ol++)
                memcpy(&out_layer[ol][offsi_base-stride_in], &lb[(y-1)%2][ol*w_out], w_out*sizeof(int32_t));
        }
    } // y
    // flush final line.
    for (int ol = 0; ol < n_out; ol++)
        memcpy(&out_layer[ol][offsi_base-stride_in], &lb[(h_out-1)%2][ol*w_out], w_out*sizeof(int32_t));
}

// instantiate int32_t and float
template
void SYN_NAME<int32_t>(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int32_t *line_buffer, int residue, int relu);
template
void SYN_NAME<float>(int KS, float *kw, float *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, float *in, int N_OUT, float *out, float *line_buffer, int residue, int relu);

#undef tostr
#undef xstr
#undef out_layer
#undef SYN_NAME
#undef SYN_KS
#undef SYN_N_IN
#undef SYN_N_OUT
#undef SYN_ATATIME
#undef SYN_RES
