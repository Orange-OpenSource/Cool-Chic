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

// stride and plane_stride are assumed the same for in and out.
// dedicated to 3x3 kernels that use a line-buffer for temporary storage, allowing in-place convolution.
void SYN_NAME(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int32_t *line_buffer, int residue, int relu)
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

    int32_t *in_layer[n_in];
    in_layer[0] = in;
    for (int i = 1; i < n_in; i++)
        in_layer[i] = in_layer[i-1]+plane_stride_in;

    int32_t *out_layer[n_out];
    out_layer[0] = out;
    for (int i = 1; i < n_out; i++)
        out_layer[i] = out_layer[i-1]+plane_stride_in;

    // in-place, must have line buffer.
    int32_t *lb[2]; // two line buffer pointers.
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

    // here we collect the output during processing, and flush later.

    // we want to operate the out kernels one after the other, using the same inputs.
    // then advance all the outs.
    int offs0 = 0;
    for (int y = 0; y < h_out; y += kstride, offs0 += stride_in)
    {
        int offs = offs0;
        int offso = offs0;
        for (int x = 0; x < w_out; x += kstride, offs += kstride, offso++)
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
                lb[y%2][ol*w_out+x] = sum;
            }
        }
        // we are at the end of this line, flush previous line buffer, if any.
        if (y >= 1)
        {
            for (int ol = 0; ol < n_out; ol++)
                memcpy(&out_layer[ol][offs0-stride_in], &lb[(y-1)%2][ol*w_out], w_out*sizeof(int32_t));
        }
    }
    // flush final line.
    for (int ol = 0; ol < n_out; ol++)
        memcpy(&out_layer[ol][offs0-stride_in], &lb[(h_out-1)%2][ol*w_out], w_out*sizeof(int32_t));
}

#undef tostr
#undef xstr
#undef out_layer
#undef SYN_NAME
#undef SYN_KS
#undef SYN_N_IN
#undef SYN_N_OUT
