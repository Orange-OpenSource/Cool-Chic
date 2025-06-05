
// defines expected:
// KS
// UPSNAME

#define tostr(x) #x
#define xstr(x) tostr(x)

// ups_src_precision is always ARM_PRECISION (from arm src)
// tmp is guaranteed to hold a horizontally upsampled in.
template <typename P>
void UPSNAME(int ks_param, P *kw, frame_memory<P> &in, frame_memory<P> &out, int ups_src_precision, frame_memory<P> &tmp)
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
    out.update_to(in.h, in.w, out.pad, out.planes); // keep pad and planes -- this could be targeting a syn output buffer.
    // pad input.
    in.custom_pad_zero_plane_in_place_i(0, pad, 0); // LR pad.

    // prepare temporary to hold h x w
    tmp.update_to(in.h, in.w, tmp.pad, tmp.planes);
    P *src = in.pad_origin(0, pad, 0);
    P *dst = tmp.origin();

    // h (horizontal treatment) to temporary.
    for (int y = 0; y < in.h; y++, src += in.stride-in.w, dst += tmp.stride-in.w)
    {
        for (int x = 0; x < in.w; x++, src++, dst++)
        {
#if 1
            // ignore symmetric.
            P sum = 0;
            for (int xx = 0; xx < ks; xx++)
                sum += src[xx]*kw[xx];
#endif
#if 0
            // use symmetric.
            P sum = src[kshalf]*kw[kshalf];
            for (int xx = 0; xx < kshalf; xx++)
                sum += (src[xx]+src[ks-xx-1])*kw[xx];
#endif
            if constexpr(std::is_same<P, int32_t>::value)
            {
                if (sum < 0)
                    *dst = -(-sum >> ups_src_precision);
                else
                    *dst = sum >> ups_src_precision;
            }
            else
            {
                *dst = sum;
            }
        }
    }

    // v (vertical treatment) to output.
    tmp.custom_pad_zero_plane_in_place_i(0, 0, pad); // TB pad.
    P *hsrc = tmp.pad_origin(0, 0, pad);
    src = in.origin();
    P *final_dst = out.origin();
#if 0
    P final_mul; // for float.
    if constexpr(std::is_same<P, float>::value)
    {
        final_mul = 1.0/(1<<UPS_PRECISION);
    }
#endif
    int residue_shift = UPS_PRECISION-ups_src_precision; // src is arm output at lower precision.
    for (int y = 0; y < in.h; y++, hsrc += tmp.stride-in.w, final_dst += out.stride-in.w, src += in.stride-in.w)
    {
        for (int x = 0; x < in.w; x++, hsrc++, final_dst++, src++)
        {
            P sum = 0;
            for (int yy = 0; yy < ks; yy++)
                sum += hsrc[yy*tmp.stride]*kw[yy];
            if constexpr(std::is_same<P, int32_t>::value)
            {
                sum += (*src<<residue_shift) << UPS_PRECISION;
                if (sum < 0)
                    *final_dst = -(-sum >> UPS_PRECISION);
                else
                    *final_dst = sum >> UPS_PRECISION;
            }
            else
            {
                *final_dst = sum+(*src);
            }
#if 0
            sum += (*src<<residue_shift) << UPS_PRECISION;
            if constexpr(std::is_same<P, int32_t>::value)
            {
                if (sum < 0)
                    *final_dst = -(-sum >> UPS_PRECISION);
                else
                    *final_dst = sum >> UPS_PRECISION;
            }
            else
            {
                *final_dst = sum*final_mul;
            }
#endif
        }
    }
}

// instantiate int32_t and float.
template
void UPSNAME<int32_t>(int ks_param, int32_t *kw, frame_memory<int32_t> &in, frame_memory<int32_t> &out, int ups_src_precision, frame_memory<int32_t> &tmp);
template
void UPSNAME<float>(int ks_param, float *kw, frame_memory<float> &in, frame_memory<float> &out, int ups_src_precision, frame_memory<float> &tmp);

#undef KS
#undef UPSNAME
#undef tostr
#undef xstr
