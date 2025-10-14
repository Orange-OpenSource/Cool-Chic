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
// SYN_N_HIDDEN
// SYN_N_OUT

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

    if (KS != ks)
    {
        printf("%s: bad call, ks=%d must be 1\n", xstr(SYN_NAME), KS);
        exit(1);
    }
    P *src = in;
    P *dst = out;
    P elements_7[n_in]; // in
    P elements_40[n_hidden]; // hidden

    for (int y = 0; y < h_in; y++, src += stride_in-w_in, dst += stride_in-w_in)
    for (int x = 0; x < w_in; x++, src++, dst++)
    {
        P *inputs = &elements_7[0];
        P *hidden = &elements_40[0];

        // load input.
        for (int i = 0; i < n_in; i++)
        {
            inputs[i] = src[i*plane_stride_in]; // gather
        }

        // ONE HIDDEN LAYER
        {
            P *kw = kw7_40;
            P *kb = kb40;

            for (int i = 0; i < n_hidden; i++)
                hidden[i] = kb[i];

            for (int il = 0; il < n_in; il++, kw += n_hidden)
            {
                for (int i = 0; i < n_hidden; i++)
                {
                    hidden[i] += inputs[il]*(kw[i]);
                }
            }

            // always relu
            for (int i = 0; i < n_hidden; i++)
            {
                if (hidden[i] < 0)
                    hidden[i] = 0;
                else
                {
                    if constexpr(std::is_same<P, int32_t>::value)
                        hidden[i] >>= SYN_MUL_PRECISION; // take multiplied sum to outputs.
                }
            }
        }

        // FINAL 40 -> 3
        P *kw = kw40_3;
        P *kb = kb3;
        for (int ol = 0; ol < n_out; ol++, kw += n_hidden)
        {
            P sum = kb[ol];
            for (int il = 0; il < n_hidden; il++)
                sum += hidden[il]*(kw[il]);
            // no relu
            if constexpr(std::is_same<P, int32_t>::value)
            {
                if (sum < 0)
                    sum = -(-sum >> SYN_MUL_PRECISION);
                else
                    sum >>= SYN_MUL_PRECISION;
            }
            dst[ol*plane_stride_in] = sum;
        }
    } // x, y
}

// instantiate int32_t and float
template
void SYN_NAME<int32_t>(int KS,
                       int32_t *kw7_40, int32_t *kb40, int32_t *kw40_3, int32_t *kb3,
                       int h_in, int w_in, int stride_in, int plane_stride_in, int N_IN, int N_HIDDEN, int32_t *in, int N_OUT, int32_t *out);
template
void SYN_NAME<float>(int KS,
                     float *kw7_40, float *kb40, float *kw40_3, float *kb3,
                     int h_in, int w_in, int stride_in, int plane_stride_in, int N_IN, int N_HIDDEN, float *in, int N_OUT, float *out);

#undef tostr
#undef xstr
#undef SYN_NAME
#undef SYN_KS
#undef SYN_N_IN
#undef SYN_N_HIDDEN
#undef SYN_N_OUT
