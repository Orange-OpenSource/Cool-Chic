/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
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

// stride and plane_stride are assumed the same for in and out.
void SYN_NAME(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu)
{
    //printf("%s(ks=%d N_IN=%d N_OUT=%d, residue=%d relu=%d\n", xstr(SYN_NAME), KS, N_IN, N_OUT, residue, relu);

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

    int32_t *in_layer[std::max(n_in, n_out)];
    in_layer[0] = in;
    for (int i = 1; i < std::max(n_in, n_out); i++)
        in_layer[i] = in_layer[i-1]+plane_stride_in;
#if SYN_KS == 1
// always in-place for ks==1
#define out_layer in_layer
    if (out != NULL && out != in)
    {
        printf("%s: ks=%d n_in=%d n_out=%d: bad call: should be in-place, but out supplied\n", xstr(SYN_NAME), KS, N_IN, N_OUT);
        exit(1);
    }
#else
    //not in-place
    int32_t *out_layer[n_out];
    out_layer[0] = out;
    for (int i = 1; i < n_out; i++)
        out_layer[i] = out_layer[i-1]+plane_stride_in;
#endif

    // here we collect the output during processing, and flush later.
    int32_t out_cache[n_out];

    // we want to operate the out kernels one after the other, using the same inputs.
    // then advance all the outs.
    int offs0 = 0;
    for (int y = 0; y < h_in-ks+1; y += kstride, offs0 += stride_in)
    {
        int offs = offs0;
        int offso = offs0;
        for (int x = 0; x < w_in-ks+1; x += kstride, offs += kstride, offso++)
        {
            int32_t *k = kw;
            for (int ol = 0; ol < n_out; ol++)
            {
                int32_t sum = (int32_t)kb[ol];
                if (residue)
                {
                    // a residue has not been multiplied.
                    sum += (int32_t)in_layer[ol][offso+residue_origin_offset]<<SYN_MUL_PRECISION;
                }
                for (int il = 0; il < n_in; il++)
                {
                    int offs2 = offs;
                    for (int yy = 0; yy < ks; yy++, offs2 += stride_in-ks)
                        for (int xx = 0; xx < ks; xx++, offs2++)
                        {
                            int32_t xxres = ((int32_t)in_layer[il][offs2]*(*k++));
                            sum += xxres;
                        }
                }
                // take multiplied sum to output after reluing.
                if (sum < 0)
                {
                    if (relu)
                        sum = 0;
                    else
                        sum = -(-sum >> SYN_MUL_PRECISION);
                }
                else
                    sum >>= SYN_MUL_PRECISION;
                out_cache[ol] = sum;
            }
            // flush.
            for (int ol = 0; ol < n_out; ol++)
            {
                out_layer[ol][offso] = out_cache[ol];
            }
        }
    }
}

#undef tostr
#undef xstr
#undef out_layer
#undef SYN_NAME
#undef SYN_KS
#undef SYN_N_IN
#undef SYN_N_OUT
