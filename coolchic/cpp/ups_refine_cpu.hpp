
// defines expected:
// KS
// UPSNAME

#define tostr(x) #x
#define xstr(x) tostr(x)

// ups_src_precision is always ARM_PRECISION (from arm src)
// tmp is guaranteed to hold a horizontally upsampled in.
void UPSNAME(int ks_param, int32_t *kw, frame_memory &in, frame_memory &out, int ups_src_precision, frame_memory &tmp)
{
#ifdef KS
    int const ks = KS;
    if (ks != ks_param)
    {
        printf("%s: bad call, ks_param %d\n", xstr(UPSNAME), ks_param);
        exit(1);
    }
#else
    int const ks = ks_param;
#endif
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

    // h (horizontal treatment) to temporary.
    for (int y = 0; y < in.h; y++, src += in.stride-in.w, dst += tmp.stride-in.w)
    {
        for (int x = 0; x < in.w; x++, src++, dst++)
        {
#if 1
            // ignore symmetric.
            int sum = 0;
            for (int xx = 0; xx < ks; xx++)
                sum += src[xx]*kw[xx];
#endif
#if 0
            // use symmetric.
            int sum = src[kshalf]*kw[kshalf];
            for (int xx = 0; xx < kshalf; xx++)
                sum += (src[xx]+src[ks-xx-1])*kw[xx];
#endif
            if (sum < 0)
                *dst = -(-sum >> ups_src_precision);
            else
                *dst = sum >> ups_src_precision;
        }
    }

    // v (vertical treatment) to output.
    tmp.custom_pad_zero_plane_in_place_i(0, 0, pad); // TB pad.
    int32_t *hsrc = tmp.pad_origin(0, 0, pad);
    src = in.origin();
    dst = out.origin();
    int residue_shift = UPS_PRECISION-ups_src_precision; // src is arm output at lower precision.
    for (int y = 0; y < in.h; y++, hsrc += tmp.stride-in.w, dst += out.stride-in.w, src += in.stride-in.w)
    {
        for (int x = 0; x < in.w; x++, hsrc++, dst++, src++)
        {
            int sum = 0;
            for (int yy = 0; yy < ks; yy++)
                sum += hsrc[yy*tmp.stride]*kw[yy];
            sum += (*src<<residue_shift) << UPS_PRECISION;
            if (sum < 0)
                *dst = -(-sum >> UPS_PRECISION);
            else
                *dst = sum >> UPS_PRECISION;
        }
    }
}

#undef KS
#undef UPSNAME
#undef tostr
#undef xstr
