
// defines expected:
// KS
// UPSNAME

#define tostr(x) #x
#define xstr(x) tostr(x)

// ups_src_precision is either ARM_PRECISION (from arm src, unrefined lowest layer) or UPS_PRECISION (from ups refinement src)
// incoming kernel is actually two filters, interleavead, to produce two outputs.
// tmp is guaranteed to hold a horizontally upsampled in.
template <typename P>
void UPSNAME(int ksx2_param, P *kw, frame_memory<P> &in, frame_memory<P> &out, int out_plane, int ups_src_precision, frame_memory<P> &tmp)
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

    P kw_even[ks];
    P kw_odd[ks];
    for (int i = 0; i < ks; i++)
    {
        kw_even[i] = kw[i*2];
        kw_odd[i] = kw[i*2+1];
    }

    // prepare temporary to hold h x 2*w
    tmp.update_to(in.h, 2*in.w, tmp.pad, tmp.planes);
    P *src = in.pad_origin(0, pad, 0);
    P *dst = tmp.origin();

    // h (horizontal scale) to temporary.
    for (int y = 0; y < in.h; y++, src += in.stride-in.w, dst += tmp.stride-2*in.w)
    {
        for (int x = 0; x < in.w; x++)
        {
            // even
            P sum_even = 0;
            P sum_odd = 0;
            for (int xx = 0; xx < ks; xx++)
            {
                sum_even += src[xx]*kw_even[xx];
                sum_odd += src[xx+1]*kw_odd[xx]; // !!! strange we need a +1 for odd.
            }
            if constexpr(std::is_same<P, int32_t>::value)
            {
                if (sum_even < 0)
                    *dst++ = -(-sum_even >> ups_src_precision);
                else
                    *dst++ = sum_even >> ups_src_precision;
                if (sum_odd < 0)
                    *dst++ = -(-sum_odd >> ups_src_precision);
                else
                    *dst++ = sum_odd >> ups_src_precision;
            }
            else
            {
                *dst++ = sum_even;
                *dst++ = sum_odd;
            }
            src++;
        }
    }

    // v (vertical) to output.
    tmp.custom_pad_replicate_plane_in_place_i(0, 0, pad); // TB pad.
    src = tmp.pad_origin(0, 0, pad);
    P *final_dst = out.plane_origin(out_plane);
#if 0
    P final_mul; // for converting int to float.
    if constexpr(std::is_same<P, float>::value)
    {
        final_mul = 1.0/(1<<UPS_PRECISION);
    }
#endif
    for (int y = 0; y < out.h; y += 2, src += tmp.stride-out.w, final_dst += out.stride-out.w+out.stride)
    {
        for (int x = 0; x < out.w; x++, src++, final_dst++)
        {
            P sum_even = 0;
            P sum_odd = 0;
            for (int yy = 0; yy < ks; yy++)
            {
                sum_even += src[yy*tmp.stride]*kw_even[yy];
                sum_odd += src[(yy+1)*tmp.stride]*kw_odd[yy]; // !!! strange we need a +1 for odd.
            }
            if constexpr(std::is_same<P, int32_t>::value)
            {
                if (sum_even < 0)
                    final_dst[0] = -(-sum_even >> UPS_PRECISION);
                else
                    final_dst[0] = sum_even >> UPS_PRECISION;
                if (sum_odd < 0)
                    final_dst[0+out.stride] = -(-sum_odd >> UPS_PRECISION);
                else
                    final_dst[0+out.stride] = sum_odd >> UPS_PRECISION;
            }
            else
            {
                final_dst[0] = sum_even;
                final_dst[0+out.stride] = sum_odd;
#if 0
                final_dst[0] = sum_even*final_mul;
                final_dst[0+out.stride] = sum_odd*final_mul;
#endif
            }
        }
    }
}

// instantiate int32_t and float.
template
void UPSNAME<int32_t>(int ksx2, int32_t *kw, frame_memory<int32_t> &in, frame_memory<int32_t> &out, int out_plane, int ups_src_precision, frame_memory<int32_t> &tmp);
template
void UPSNAME<float>(int ksx2, float *kw, frame_memory<float> &in, frame_memory<float> &out, int out_plane, int ups_src_precision, frame_memory<float> &tmp);

#undef KS
#undef UPSNAME
#undef tostr
#undef xstr
