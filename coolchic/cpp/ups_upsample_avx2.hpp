
// defines expected:
// KS
// optional UPS_SRC_PRECISION
// UPSNAME


#define tostr(x) #x
#define xstr(x) tostr(x)

// ups_src_precision is either ARM_PRECISION (from arm src, unrefined lowest layer) or UPS_PRECISION (from ups refinement src)
// incoming kernel is actually two filters, interleavead, to produce two outputs.
// tmp is guaranteed to hold a horizontally upsampled in.
void UPSNAME(int ksx2_param, int32_t *kw, frame_memory &in, frame_memory &out, int out_plane, int ups_src_precision, frame_memory &tmp)
{
#ifdef KS
    int const ksx2 = KS;
    if (ksx2 != ksx2_param)
    {
        printf("%s: bad call, ksx2_param %d\n", xstr(UPSNAME), ksx2_param);
        exit(1);
    }
#else
    int const ksx2 = ksx2_param;
#endif
    int const ks = ksx2/2; // 2 kernels of size ks.
    int pad = ks/2; // incoming ks is two even-sized filters.
    in.custom_pad_replicate_plane_in_place_i(0, pad, 0); // LR pad.

#ifdef UPS_SRC_PRECISION
    if (ups_src_precision != UPS_SRC_PRECISION)
    {
        printf("%s: bad call: input precision is %d, not %d\n", xstr(UPSNAME), ups_src_precision, UPS_SRC_PRECISION);
        exit(1);
    }
#endif

#ifdef KS
    // kernel in registers
    __m256i KW_EVEN[KS];
    __m256i KW_ODD[KS];
    for (int i = 0; i < KS; i++)
    {
        KW_EVEN[i] = _mm256_set1_epi32(kw[i*2]);
        KW_ODD[i] = _mm256_set1_epi32(kw[i*2+1]);
    }
#else
    int32_t kw_even[ks];
    int32_t kw_odd[ks];
    for (int i = 0; i < ks; i++)
    {
        kw_even[i] = kw[i*2];
        kw_odd[i] = kw[i*2+1];
    }
#endif

    // prepare temporary to hold h x 2*w
    tmp.update_to(in.h, 2*in.w, tmp.pad, tmp.planes);
    int32_t *src = in.pad_origin(0, pad, 0);
    int32_t *dst = tmp.origin();
    const __m256i z = _mm256_setzero_si256();

    // h (horizontal scale) to temporary.
    int xlim_blk = (in.w+7)/8; // pixels in number of horizontal blocks
    int xlast_blk_size = 8-(xlim_blk*8 - in.w); // number of pixels to emit per filter in last block.
    int32_t store[8];

    for (int y = 0; y < in.h; y++, src += in.stride-xlim_blk*8, dst += tmp.stride-xlim_blk*8*2)
    {
        for (int x_blk = 0; x_blk < xlim_blk; x_blk++, src += 8, dst += 8*2)
        {
            int n_outs = (x_blk < xlim_blk-1) ? 8 : xlast_blk_size;
            __m256i out_even_avx2 = _mm256_setzero_si256();
            __m256i out_odd_avx2 = _mm256_setzero_si256();
            for (int xx = 0; xx < ks; xx++)
            {
                __m256i input_even = _mm256_loadu_si256((__m256i_u*)&src[xx]);
                __m256i input_odd = _mm256_loadu_si256((__m256i_u*)&src[xx+1]); // !!! strange we need a +1 for odd.
#ifdef KS
                __m256i mul = _mm256_mullo_epi32(input_even, KW_EVEN[xx]);
                out_even_avx2 = _mm256_add_epi32(out_even_avx2, mul);
                mul = _mm256_mullo_epi32(input_odd, KW_ODD[xx]);
                out_odd_avx2 = _mm256_add_epi32(out_odd_avx2, mul);
#else
                __m256i kk  = _mm256_set1_epi32(kw_even[xx]);
                __m256i mul = _mm256_mullo_epi32(input_even, kk);
                out_even_avx2 = _mm256_add_epi32(out_even_avx2, mul);
                kk  = _mm256_set1_epi32(kw_odd[xx]);
                mul = _mm256_mullo_epi32(input_odd, kk);
                out_odd_avx2 = _mm256_add_epi32(out_odd_avx2, mul);
#endif
            }
            // need different treatment for -ve -(-(>>)) and +ve (>>).
#ifdef UPS_SRC_PRECISION
            __m256i sr  = _mm256_srai_epi32(out_even_avx2, UPS_SRC_PRECISION);
            __m256i negsrneg = _mm256_sub_epi32(z, _mm256_srai_epi32(_mm256_sub_epi32(z, out_even_avx2), UPS_SRC_PRECISION));
            out_even_avx2 = _mm256_blendv_epi8(negsrneg, sr, _mm256_cmpgt_epi32(out_even_avx2, z));

            sr  = _mm256_srai_epi32(out_odd_avx2, UPS_SRC_PRECISION);
            negsrneg = _mm256_sub_epi32(z, _mm256_srai_epi32(_mm256_sub_epi32(z, out_odd_avx2), UPS_SRC_PRECISION));
            out_odd_avx2 = _mm256_blendv_epi8(negsrneg, sr, _mm256_cmpgt_epi32(out_odd_avx2, z));
#else
            __m256i sr  = _mm256_srai_epi32(out_even_avx2, ups_src_precision);
            __m256i negsrneg = _mm256_sub_epi32(z, _mm256_srai_epi32(_mm256_sub_epi32(z, out_even_avx2), ups_src_precision));
            out_even_avx2 = _mm256_blendv_epi8(negsrneg, sr, _mm256_cmpgt_epi32(out_even_avx2, z));

            sr  = _mm256_srai_epi32(out_odd_avx2, ups_src_precision);
            negsrneg = _mm256_sub_epi32(z, _mm256_srai_epi32(_mm256_sub_epi32(z, out_odd_avx2), ups_src_precision));
            out_odd_avx2 = _mm256_blendv_epi8(negsrneg, sr, _mm256_cmpgt_epi32(out_odd_avx2, z));
#endif

            // spread horizontally
            __m256i outA0 = _mm256_unpacklo_epi32(out_even_avx2, out_odd_avx2);
            __m256i outA1 = _mm256_unpackhi_epi32(out_even_avx2, out_odd_avx2);
            // We need to remangle A0,A1 switching their high and low 128-bit halves.
            out_even_avx2 = _mm256_permute2f128_si256(outA0, outA1, 0x20);
            out_odd_avx2 = _mm256_permute2f128_si256(outA0, outA1, 0x31);
            if (n_outs == 8)
            {
                _mm256_storeu_si256((__m256i_u*)(dst+0), out_even_avx2);
                _mm256_storeu_si256((__m256i_u*)(dst+8), out_odd_avx2);
            }
            else
            {
                n_outs *= 2;
                if (n_outs >= 8)
                {
                    _mm256_storeu_si256((__m256i_u*)(dst+0), out_even_avx2);
                    if (n_outs >= 16)
                        _mm256_storeu_si256((__m256i_u*)(dst+8), out_odd_avx2);
                    else if (n_outs > 8)
                    {
                        // remainder over 8.
                        _mm256_storeu_si256((__m256i_u*)&store[0], out_odd_avx2);
                        memcpy(&dst[8], &store[0], (n_outs-8)*sizeof(store[0]));
                    }
                }
                else
                {
                    // remainder over 0.
                    _mm256_storeu_si256((__m256i_u*)&store[0], out_even_avx2);
                    memcpy(&dst[0], &store[0], n_outs*sizeof(store[0]));
                }
            }
        }
    }

    // v (vertical) to output.
    tmp.custom_pad_replicate_plane_in_place_i(0, 0, pad); // TB pad.
    int32_t *hsrc = tmp.pad_origin(0, 0, pad);
    dst = out.plane_origin(out_plane);

    xlim_blk = (tmp.w+7)/8; // pixels in number of horizontal blocks
    xlast_blk_size = 8-(xlim_blk*8 - tmp.w); // number of pixels to emit per filter in last block.

    for (int y = 0; y < out.h; y +=2, hsrc += tmp.stride-xlim_blk*8, dst += out.stride-xlim_blk*8+out.stride)
    {
        for (int x_blk = 0; x_blk < xlim_blk; x_blk++, hsrc += 8, dst += 8)
        {
            int n_outs = (x_blk < xlim_blk-1) ? 8 : xlast_blk_size;
            __m256i out_even_avx2 = _mm256_setzero_si256();
            __m256i out_odd_avx2 = _mm256_setzero_si256();
            for (int yy = 0; yy < ks; yy++)
            {
                __m256i input_even = _mm256_loadu_si256((__m256i_u*)&hsrc[yy*tmp.stride]);
                __m256i input_odd = _mm256_loadu_si256((__m256i_u*)&hsrc[(yy+1)*tmp.stride]); // !!! strange we need a +1 for odd.
#ifdef KS
                __m256i mul = _mm256_mullo_epi32(input_even, KW_EVEN[yy]);
                out_even_avx2 = _mm256_add_epi32(out_even_avx2, mul);
                mul = _mm256_mullo_epi32(input_odd, KW_ODD[yy]);
                out_odd_avx2 = _mm256_add_epi32(out_odd_avx2, mul);
#else
                __m256i kk  = _mm256_set1_epi32(kw_even[yy]);
                __m256i mul = _mm256_mullo_epi32(input_even, kk);
                out_even_avx2 = _mm256_add_epi32(out_even_avx2, mul);
                kk  = _mm256_set1_epi32(kw_odd[yy]);
                mul = _mm256_mullo_epi32(input_odd, kk);
                out_odd_avx2 = _mm256_add_epi32(out_odd_avx2, mul);
#endif
            }
            // need different treatment for -ve -(-(>>)) and +ve (>>).
            __m256i sr  = _mm256_srai_epi32(out_even_avx2, UPS_PRECISION);
            __m256i negsrneg = _mm256_sub_epi32(z, _mm256_srai_epi32(_mm256_sub_epi32(z, out_even_avx2), UPS_PRECISION));
            out_even_avx2 = _mm256_blendv_epi8(negsrneg, sr, _mm256_cmpgt_epi32(out_even_avx2, z));

            sr  = _mm256_srai_epi32(out_odd_avx2, UPS_PRECISION);
            negsrneg = _mm256_sub_epi32(z, _mm256_srai_epi32(_mm256_sub_epi32(z, out_odd_avx2), UPS_PRECISION));
            out_odd_avx2 = _mm256_blendv_epi8(negsrneg, sr, _mm256_cmpgt_epi32(out_odd_avx2, z));

            if (n_outs == 8)
            {
                _mm256_storeu_si256((__m256i_u*)(dst+0*out.stride), out_even_avx2);
                _mm256_storeu_si256((__m256i_u*)(dst+1*out.stride), out_odd_avx2);
            }
            else
            {
                _mm256_storeu_si256((__m256i_u*)&store[0], out_even_avx2);
                memcpy(&dst[0], &store[0], n_outs*sizeof(store[0]));
                _mm256_storeu_si256((__m256i_u*)&store[0], out_odd_avx2);
                memcpy(&dst[out.stride], &store[0], n_outs*sizeof(store[0]));
            }
        }
    }
}


#undef KS
#undef UPS_SRC_PRECISION
#undef UPSNAME
#undef tostr
#undef xstr
