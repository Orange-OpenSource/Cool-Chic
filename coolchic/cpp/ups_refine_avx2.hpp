
// defines expected:
// KS
// UPSNAME

#define tostr(x) #x
#define xstr(x) tostr(x)

// ups_src_precision is always ARM_PRECISION (from arm src)
void UPSNAME(int ks_param, int32_t *kw, frame_memory &in, frame_memory &out, int ups_src_precision, frame_memory &tmp)
{
#ifdef KS
    int const ks = KS;
    if (ks != ks_param)
    {
        printf("%s: bad call, ks_param %d\n", xstr(UPSNAME), ks_param);
        exit(1);
    }

    // kernel in registers
    __m256i KW[KS];
    for (int i = 0; i < KS; i++)
        KW[i] = _mm256_set1_epi32(kw[i]);
#else
    int const ks = ks_param;
#endif
    // temporary check.
    if (ups_src_precision != ARM_PRECISION)
    {
        printf("%s: bad call: input precision is %d, not %d\n", xstr(UPSNAME), ups_src_precision, ARM_PRECISION);
        exit(1);
    }
    int const UPS_SRC_PRECISION = ARM_PRECISION;
    int kshalf = ks/2;
    int pad = kshalf;
    // prepare output.
    out.update_to(in.h, in.w, out.pad, 1);
    // pad input.
    in.custom_pad_zero_plane_in_place_i(0, pad, 0); // LR pad.

    // prepare temporary to hold h x w
    tmp.update_to(in.h, in.w, tmp.pad, tmp.planes);
    int32_t *src = in.pad_origin(0, pad, 0);
    int32_t *dst = tmp.origin();
    const __m256i z = _mm256_setzero_si256();

    // h (horizontal treatment) to temporary.
    int xlim_blk = (in.w+7)/8; // pixels in number of horizontal blocks
    int xlast_blk_size = 8-(xlim_blk*8 - in.w); // number of pixels to emit per filter in last block.
    int32_t store[8];

    for (int y = 0; y < in.h; y++, src += in.stride-xlim_blk*8, dst += tmp.stride-xlim_blk*8)
    {
        for (int x_blk = 0; x_blk < xlim_blk; x_blk++, src += 8, dst += 8)
        {
            int n_outs = (x_blk < xlim_blk-1) ? 8 : xlast_blk_size;
            __m256i out_avx2 = _mm256_setzero_si256();

            for (int xx = 0; xx < ks; xx++)
            {
                __m256i input = _mm256_loadu_si256((__m256i_u*)&src[xx]);
#ifdef KS
                __m256i mul = _mm256_mullo_epi32(input, KW[xx]);
#else
                __m256i kk  = _mm256_set1_epi32(kw[xx]);
                __m256i mul = _mm256_mullo_epi32(input, kk);
#endif
                out_avx2 = _mm256_add_epi32(out_avx2, mul);
            }
            // need different treatment for -ve -(-(>>)) and +ve (>>).
            __m256i sr  = _mm256_srai_epi32(out_avx2, ARM_PRECISION);
            __m256i negsrneg = _mm256_sub_epi32(z, _mm256_srai_epi32(_mm256_sub_epi32(z, out_avx2), ARM_PRECISION));
            out_avx2 = _mm256_blendv_epi8(negsrneg, sr, _mm256_cmpgt_epi32(out_avx2, z));

            if (n_outs == 8)
            {
                _mm256_storeu_si256((__m256i_u*)dst, out_avx2);
            }
            else
            {
                _mm256_storeu_si256((__m256i_u*)&store[0], out_avx2);
                memcpy(dst, &store[0], n_outs*sizeof(store[0]));
            }
        }
    }

    // v (vertical treatment) to output.
    tmp.custom_pad_zero_plane_in_place_i(0, 0, pad); // TB pad.
    int32_t *hsrc = tmp.pad_origin(0, 0, pad);
    src = in.origin();
    dst = out.origin();
    int residue_shift = UPS_PRECISION-UPS_SRC_PRECISION; // src is arm output at lower precision.

    for (int y = 0; y < in.h; y++, hsrc += tmp.stride-xlim_blk*8, src += in.stride-xlim_blk*8, dst += out.stride-xlim_blk*8)
    {
        for (int x_blk = 0; x_blk < xlim_blk; x_blk++, hsrc += 8, src += 8, dst += 8)
        {
            int n_outs = (x_blk < xlim_blk-1) ? 8 : xlast_blk_size;
            __m256i out_avx2 = _mm256_loadu_si256((__m256i_u*)&src[0]);
            out_avx2 = _mm256_slli_epi32(out_avx2, residue_shift+UPS_PRECISION);

            for (int yy = 0; yy < ks; yy++)
            {
                __m256i input = _mm256_loadu_si256((__m256i_u*)&hsrc[yy*tmp.stride]);
#ifdef KS
                __m256i mul = _mm256_mullo_epi32(input, KW[yy]);
#else
                __m256i kk  = _mm256_set1_epi32(kw[yy]);
                __m256i mul = _mm256_mullo_epi32(input, kk);
#endif
                out_avx2 = _mm256_add_epi32(out_avx2, mul);
            }

            // need different treatment for -ve -(-(>>)) and +ve (>>).
            __m256i sr  = _mm256_srai_epi32(out_avx2, UPS_PRECISION);
            __m256i negsrneg = _mm256_sub_epi32(z, _mm256_srai_epi32(_mm256_sub_epi32(z, out_avx2), UPS_PRECISION));
            out_avx2 = _mm256_blendv_epi8(negsrneg, sr, _mm256_cmpgt_epi32(out_avx2, z));

            if (n_outs == 8)
            {
                _mm256_storeu_si256((__m256i_u*)dst, out_avx2);
            }
            else
            {
                _mm256_storeu_si256((__m256i_u*)&store[0], out_avx2);
                memcpy(dst, &store[0], n_outs*sizeof(store[0]));
            }
        }
    }
}

#undef KS
#undef UPSNAME
#undef tostr
#undef xstr
