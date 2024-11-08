
// out = out + in*blend_val
void syn_blend1(int h_in, int w_in, int stride_in, int plane_stride_in, int N_INOUT, int32_t *in, int32_t blend_val, int32_t *out)
{
    for (int p = 0; p < N_INOUT; p++)
    {
        int32_t *src = in+plane_stride_in*p;
        int32_t *dst = out+plane_stride_in*p;
        for (int y = 0; y < h_in; y++, src += stride_in-w_in, dst += stride_in-w_in)
        {
            for (int x = 0; x < w_in; x++, src++, dst++)
            {
                int x0 = *src;
                if (x0 < 0)
                    x0 = 0;
                else if (x0 > (1<<SYN_LAYER_PRECISION))
                    x0 = 1<<SYN_LAYER_PRECISION;
                *dst = *dst + (((x0)*blend_val) >> SYN_LAYER_PRECISION);
            }
        }
    }
}

// out = out*blend_val_out + in*blend_val_in
void syn_blend2(int h_in, int w_in, int stride_in, int plane_stride_in, int N_INOUT, int32_t *in, int32_t blend_val_in, int32_t *out, int32_t blend_val_out)
{
    for (int p = 0; p < N_INOUT; p++)
    {
        int32_t *src = in+plane_stride_in*p;
        int32_t *dst = out+plane_stride_in*p;
        for (int y = 0; y < h_in; y++, src += stride_in-w_in, dst += stride_in-w_in)
        {
            for (int x = 0; x < w_in; x++, src++, dst++)
            {
                int x0 = *src;
                if (x0 < 0)
                    x0 = 0;
                else if (x0 > (1<<SYN_LAYER_PRECISION))
                    x0 = 1<<SYN_LAYER_PRECISION;
                int x1 = *dst;
                if (x1 < 0)
                    x1 = 0;
                else if (x1 > (1<<SYN_LAYER_PRECISION))
                    x1 = 1<<SYN_LAYER_PRECISION;

                *dst = (((x1)*blend_val_out) + ((x0)*blend_val_in)) >> SYN_LAYER_PRECISION;
            }
        }
    }
}
