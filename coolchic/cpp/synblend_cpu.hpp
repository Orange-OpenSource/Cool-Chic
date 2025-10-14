/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

// out = out + in*blend_val
template<typename P>
void syn_blend1(int h_in, int w_in, int stride_in, int plane_stride_in, int N_INOUT, P *in, int32_t blend_val, P *out)
{
    P bval;
    if constexpr(std::is_same<P, int32_t>::value)
        bval = blend_val;
    else
        bval = blend_val / ((1<<SYN_LAYER_PRECISION)+0.0);
    for (int p = 0; p < N_INOUT; p++)
    {
        P *src = in+plane_stride_in*p;
        P *dst = out+plane_stride_in*p;
        for (int y = 0; y < h_in; y++, src += stride_in-w_in, dst += stride_in-w_in)
        {
            for (int x = 0; x < w_in; x++, src++, dst++)
            {
                P x0 = *src;
                if constexpr(std::is_same<P, int32_t>::value)
                {
                    if (x0 < 0)
                        x0 = 0;
                    else if (x0 > (1<<SYN_LAYER_PRECISION))
                        x0 = (1<<SYN_LAYER_PRECISION);
                    *dst = *dst + (((x0)*bval) >> SYN_LAYER_PRECISION);
                }
                else
                {
                    if (x0 < 0)
                        x0 = 0;
                    else if (x0 > 1.0)
                        x0 = 1.0;
                    *dst = *dst + ((x0)*bval);
                }
            }
        }
    }
}

// out = out*blend_val_out + in*blend_val_in
template <typename P>
void syn_blend2(int h_in, int w_in, int stride_in, int plane_stride_in, int N_INOUT, P *in, int32_t blend_val_in, P *out, int32_t blend_val_out)
{
    P bval_in;
    P bval_out;
    if constexpr(std::is_same<P, int32_t>::value)
    {
        bval_in = blend_val_in;
        bval_out = blend_val_out;
    }
    else
    {
        bval_in = blend_val_in/((1<<SYN_LAYER_PRECISION)+0.0);
        bval_out = blend_val_out/((1<<SYN_LAYER_PRECISION)+0.0);
    }
    for (int p = 0; p < N_INOUT; p++)
    {
        P *src = in+plane_stride_in*p;
        P *dst = out+plane_stride_in*p;
        for (int y = 0; y < h_in; y++, src += stride_in-w_in, dst += stride_in-w_in)
        {
            for (int x = 0; x < w_in; x++, src++, dst++)
            {
                P x0 = *src;
                P x1 = *dst;
                if constexpr(std::is_same<P, int32_t>::value)
                {
                    if (x0 < 0)
                        x0 = 0;
                    else if (x0 > (1<<SYN_LAYER_PRECISION))
                        x0 = 1<<SYN_LAYER_PRECISION;
                    if (x1 < 0)
                        x1 = 0;
                    else if (x1 > 1<<SYN_LAYER_PRECISION)
                        x1 = 1<<SYN_LAYER_PRECISION;
                    *dst = (((x1)*blend_val_out) + ((x0)*blend_val_in)) >> SYN_LAYER_PRECISION;
                }
                else
                {
                    if (x0 < 0)
                        x0 = 0;
                    else if (x0 > 1.0)
                        x0 = 1.0;
                    if (x1 < 0)
                        x1 = 0;
                    else if (x1 > 1.0)
                        x1 = 1.0;
                    *dst = ((x1)*bval_out) + ((x0)*bval_in);
                }
            }
        }
    }
}

// instantiate int32_t and float
template
void syn_blend1<int32_t>(int h_in, int w_in, int stride_in, int plane_stride_in, int N_INOUT, int32_t *in, int32_t blend_val, int32_t *out);
template
void syn_blend1<float>(int h_in, int w_in, int stride_in, int plane_stride_in, int N_INOUT, float *in, int32_t blend_val, float *out);

template
void syn_blend2<int32_t>(int h_in, int w_in, int stride_in, int plane_stride_in, int N_INOUT, int32_t *in, int32_t blend_val_in, int32_t *out, int32_t blend_val_out);
template
void syn_blend2<float>(int h_in, int w_in, int stride_in, int plane_stride_in, int N_INOUT, float *in, int32_t blend_val_in, float *out, int32_t blend_val_out);
