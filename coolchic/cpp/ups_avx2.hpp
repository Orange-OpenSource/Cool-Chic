/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#define tostr(x) #x
#define xstr(x) tostr(x)

// kw: weights for 4 4x4 kernels.
void SYN_NAME(int KS, float *kw, int h_in, int w_in, float *in, float *out)
{
    //printf("%s(ks=%d N_IN=%d N_OUT=%d, residue=%d, relu=%d\n", xstr(SYN_NAME), KS, N_IN, N_OUT, residue, relu);
    int const kstride = 1;
#ifdef SYN_KS
    int const ks = SYN_KS;
#else
    int const ks = KS;
#endif
    int const n_out = 4; // number of kernels, actually, always 4.

    int w_out = 2*(w_in-ks+1);
    if (ks != KS)
    {
        printf("%s: ks=%d: bad call\n", xstr(SYN_NAME), KS);
        exit(1);
    }

    int const ks2 = ks*ks;

    // we generate 8 outputs at a time, per kernel.
    // these outputs are emitted onto two output lines.
    // we advance 8 pixels in the input after that.
    int offsi_base = 0;
    int offso = 0;
    // the last block in x might not be full.

    int xlim_blk = ((w_in-ks+1)+7)/8; // pixels in number of horizontal blocks
    int xlast_blk_size = 8-(xlim_blk*8 - (w_in-ks+1)); // number of pixels to emit per filter in last block.
    //printf("out-pixel-blks %d last-blk-size %d giving output %d (expected %d)\n", xlim_blk, xlast_blk_size, 2*xlim_blk*8-16+2*xlast_blk_size, w_out);
    for (int y = 0; y < h_in-ks+1; y += kstride, offsi_base += w_in, offso += w_out)
    {
        int offsi_line = offsi_base;
        for (int x_blk = 0; x_blk < xlim_blk; x_blk += kstride, offsi_line += kstride*8)
        {
            __m256 out_avx2[n_out];
            for (int ol = 0; ol < n_out; ol++)
            {
                out_avx2[ol] = _mm256_setzero_ps();
            }
            int offsi_block = offsi_line;
            int koffs = 0;
            for (int yy = 0; yy < ks; yy++, offsi_block += w_in-ks)
            {
                for (int xx = 0; xx < ks; xx++, offsi_block++, koffs++)
                {
                    __m256 input = _mm256_loadu_ps(&in[offsi_block]);
                    for (int ol = 0; ol < n_out; ol++)
                    {
                        __m256 kk = _mm256_broadcast_ss(&kw[ol*ks2+koffs]);
                        __m256 mul = _mm256_mul_ps(input, kk);
                        out_avx2[ol] = _mm256_add_ps(out_avx2[ol], mul);
                    }
                }
            }
            // merge the outputs into the destination!
            // We have:
            // ol0 A0 A1 A2 A3.. A7
            // ol1 B0 B1 B2 B3.. B7
            // ol2 C0 C1 C2 C3.. C7
            // ol3 D0 D1 D2 D3.. D7
            //
            // to go to 2 lines as:
            // A0 B0 A1 B1 A2 B2.. A7 B7
            // C0 D0 C1 D1 C2 D2.. C7 D7

            // spread horizontally and over two lines.
            float outputs[4][8];
            for (int i = 0; i < 4; i++)
                _mm256_storeu_ps(&outputs[i][0], out_avx2[i]);
            if (x_blk < xlim_blk-1)
            {
                for (int line = 0; line < 2; line++)
                {
                    for (int i = 0; i < 8; i++)
                    {
                        out[offso+line*w_out+2*i+0]   = outputs[line*2+0][i];
                        out[offso+line*w_out+2*i+1]   = outputs[line*2+1][i];
                    }
                }
                offso += 2*8;
            }
            else
            {
                int n_outs = xlast_blk_size;
                for (int line = 0; line < 2; line++)
                {
                    for (int i = 0; i < n_outs; i++)
                    {
                        out[offso+line*w_out+2*i+0]   = outputs[line*2+0][i];
                        out[offso+line*w_out+2*i+1]   = outputs[line*2+1][i];
                    }
                }
                offso += 2*n_outs;
            }
        }
    }
}

#undef tostr
#undef xstr
#undef SYN_NAME
#undef SYN_KS
