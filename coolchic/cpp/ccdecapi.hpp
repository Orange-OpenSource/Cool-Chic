/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


/*
 * USED WITH
 * #define CCDECAPI_CPU
 * or
 * #define CCDECAPI_AVX2
 */

#include <stdlib.h>
#include <iostream>
#include <memory>
#include <chrono> // timing.
#include <math.h>
#include <sstream>

#include "TDecBinCoderCABAC.h"

#include "cc-contexts.h" // Emitted from python extraction, same for all extracts.
#include "common.h"
#include "cc-bitstream.h"

// Return a table of n_contexts values indicating pixel offsets from current to context pixels.
int32_t *get_context_indicies(int n_contexts, int stride)
{
    static int32_t indicies_constant[32];
    if (n_contexts == 8)
    {
        int32_t indicies[] = {                                       -3*stride+0,
                                                                     -2*stride+0,
                                                        -1*stride-1, -1*stride+0, -1*stride+1,
                              -0*stride-3, -0*stride-2, -0*stride-1 };
        memcpy(indicies_constant, indicies, n_contexts*sizeof(indicies[0]));
    }
    else if (n_contexts == 16)
    {
        int32_t indicies[] = {                                       -3*stride+0, -3*stride+1,
                                           -2*stride-2, -2*stride-1, -2*stride+0, -2*stride+1, -2*stride+2,
                              -1*stride-3, -1*stride-2, -1*stride-1, -1*stride+0, -1*stride+1, -1*stride+2,
                              -0*stride-3, -0*stride-2, -0*stride-1 };
        memcpy(indicies_constant, indicies, n_contexts*sizeof(indicies[0]));
    }
    else if (n_contexts == 24)
    {
        int32_t indicies[] = {                                                    -4*stride+0,
                                                        -3*stride-2, -3*stride-1, -3*stride+0, -3*stride+1, -3*stride+2,
                                           -2*stride-3, -2*stride-2, -2*stride-1, -2*stride+0, -2*stride+1, -2*stride+2, -2*stride+3,
                                           -1*stride-3, -1*stride-2, -1*stride-1, -1*stride+0, -1*stride+1, -1*stride+2, -1*stride+3,
                              -0*stride-4, -0*stride-3, -0*stride-2, -0*stride-1 };
        memcpy(indicies_constant, indicies, n_contexts*sizeof(indicies[0]));
    }
    else if (n_contexts == 32)
    {
        int32_t indicies[] = {                          -4*stride-2, -4*stride-1, -4*stride+0, -4*stride+1,
                                           -3*stride-3, -3*stride-2, -3*stride-1, -3*stride+0, -3*stride+1, -3*stride+2, -3*stride+3,
                                           -2*stride-3, -2*stride-2, -2*stride-1, -2*stride+0, -2*stride+1, -2*stride+2, -2*stride+3, -2*stride+4,
                              -1*stride-4, -1*stride-3, -1*stride-2, -1*stride-1, -1*stride+0, -1*stride+1, -1*stride+2, -1*stride+3, -1*stride+4,
                              -0*stride-4, -0*stride-3, -0*stride-2, -0*stride-1 };
        memcpy(indicies_constant, indicies, n_contexts*sizeof(indicies[0]));
    }
    else
    {
        printf("n_context_pixels: can only be 8, 16, 24, 32, not %d\n", n_contexts);
        exit(1);
    }
    return indicies_constant;
}

#ifdef CCDECAPI_AVX2
#include "arm_avx2.h"
#include "ups_avx2.h"
#include "syn_avx2.h"
#include "rest_avx2.h"
#endif

#include "arm_cpu.h"
#include "syn_cpu.h"

float time_bac_seconds = 0.0;
float time_arm_seconds = 0.0;
float time_ups_seconds = 0.0;
float time_syn_seconds = 0.0;
float time_warp_seconds = 0.0;
float time_bpred_seconds = 0.0;
float time_all_seconds = 0.0;

// we are assuming there is enough space outside the image bounds for the pad.
void custom_pad_replicate_plane_in_place(int h_in, int w_in, int stride_in, float *in, int in_pad)
{
    float *scan_in = in;
    float *scan_out = in - in_pad*stride_in-in_pad;
    int w_in_padded = w_in+2*in_pad;
    // leading lines
    for (int y = 0; y < in_pad; y++)
    {
        if (y == 0)
        {
            // construct first padded line.
            for (int x = 0; x < in_pad; x++)
            {
                *scan_out++ = *scan_in;
            }
            memcpy(scan_out, scan_in, w_in*sizeof(scan_out[0]));
            scan_out += w_in;
            for (int x = 0; x < in_pad; x++)
            {
                *scan_out++ = scan_in[w_in-1];
            }
            scan_out += stride_in-w_in_padded;
        }
        else
        {
            // copy previous line.
            memcpy(scan_out, scan_out-stride_in, w_in_padded*sizeof(scan_out[0]));
            scan_out += stride_in;
        }
    }
    // internal lines: pad left and right edges.
    for (int y = 0; y < h_in; y++)
    {
        scan_out = scan_in-in_pad;
        for (int x = 0; x < in_pad; x++)
        {
            *scan_out++ = *scan_in;
        }
        scan_out += w_in;
        for (int x = 0; x < in_pad; x++)
        {
            *scan_out++ = scan_in[w_in-1];
        }
        scan_in += stride_in;
    }
    // trailing lines -- we copy the previous padded.
    scan_in -= stride_in; // beginning of last line, padding to the left.
    scan_out = scan_in+stride_in-in_pad;
    for (int y = 0; y < in_pad; y++)
    {
        memcpy(scan_out, scan_out-stride_in, w_in_padded*sizeof(scan_out[0]));
        scan_out += stride_in;
    }
}

// copies everything to pad -- used in upsampling.
void custom_pad_replicate_plane(int h_in, int w_in, float *in, int in_pad, float *out)
{
    int w_in_padded = w_in+2*in_pad;
    float *scan_in = in;
    float *scan_out = out;
    // leading lines, including first real.
    for (int y = 0; y < in_pad+1; y++) // +1 for 1st real, too.
    {
        if (y == 0)
        {
            for (int x = 0; x < in_pad; x++)
            {
                *scan_out++ = *scan_in;
            }
            memcpy(scan_out, scan_in, w_in*sizeof(scan_out[0]));
            scan_out += w_in;
            scan_in += w_in;
            for (int x = 0; x < in_pad; x++)
            {
                *scan_out++ = scan_in[-1];
            }
        }
        else
        {
            memcpy(scan_out, scan_out-w_in_padded, w_in_padded*sizeof(scan_out[0]));
            scan_out += w_in_padded;
        }
    }
    // internal lines
    for (int y = 1; y < h_in; y++) // from 1, because 1st real was last above.
    {
        for (int x = 0; x < in_pad; x++)
        {
            *scan_out++ = *scan_in;
        }
        memcpy(scan_out, scan_in, w_in*sizeof(scan_out[0]));
        scan_out += w_in;
        scan_in += w_in;
        for (int x = 0; x < in_pad; x++)
        {
            *scan_out++ = scan_in[-1];
        }
    }
    // trailing lines
    for (int y = 0; y < in_pad; y++)
    {
        memcpy(scan_out, scan_out-w_in_padded, w_in_padded*sizeof(scan_out[0]));
        scan_out += w_in_padded;
    }
}

// applying 4 small kernels, ks here is upsampling_kernel/2
void custom_conv_ups_zxz1_cpu(int ks, float *kw, int h_in, int w_in, float *in, float *out)
{
    int const stride = 1;
    int offs0 = 0;
    for (int y = 0; y < h_in-ks+1; y += stride, offs0 += w_in)
    {
        for (int vf = 0; vf < 2; vf++) // vertical filter choice on this line.
        {
            int offs = offs0;
            for (int x = 0; x < w_in-ks+1; x += stride, offs += stride)
            {
                for (int hf = 0; hf < 2; hf++) // horizontal filter choice at this point
                {
                    float sum = 0;
                    float *k = kw+(vf*2+hf)*(ks*ks);
                    int offs2 = offs;
                    for (int yy = 0; yy < ks; yy++, offs2 += w_in-ks)
                        for (int xx = 0; xx < ks; xx++)
                        {
                            //if (y == 0 && x == 0)
                            //    printf(" k%g*i%g", *k, in[offs2]);
                            sum += in[offs2++]*(*k++);
                        }
                    *out++ = sum;
                }
            }
        }
    }
}

// upsampling.
// upsxpose -- already-transposed kernel.
// out -- will be 2*h_in, 2*w_in

// Does not expand the source data, and applies 4 smaller kernels.
void custom_upsample_4(int ks, float *weightsx4xpose, int h_in, int w_in, float *in, float *out, int h_target, int w_target, int stride_target)
{
    static int big_dim = 0;
    static float *padded_ups = NULL;
    static float *in_padded = NULL;

    int this_dim = ks-1 + (std::max(h_in, w_in)+2*((ks/2)/2)-1)*2+1 +ks-1; // biggest HD dimension, padded.
    if (this_dim > big_dim)
    {
        big_dim = this_dim;
        delete[] padded_ups;
        delete[] in_padded;

        padded_ups = new float[big_dim*big_dim]; // we need to crop to 'out'
        // pad. replicated.
        // we increase w_in and h_in in a different buffer for the moment.
        in_padded = (float *)new float[big_dim*big_dim];
    }

    custom_pad_replicate_plane(h_in, w_in, in, ks/2, in_padded);

    // data on which to operate is now in_padded, with width, height h_in_padded, w_in_padded.
    int in_pad = ks/2;
    int h_in_padded = h_in+2*in_pad;
    int w_in_padded = w_in+2*in_pad;

#ifdef CCDECAPI_AVX2
    if (ks == 8)
    {
        ups_4x4x4_avx2(ks/2, weightsx4xpose, h_in_padded, w_in_padded, in_padded, padded_ups);
    }
    else if (ks == 4)
    {
        ups_4x2x2_avx2(ks/2, weightsx4xpose, h_in_padded, w_in_padded, in_padded, padded_ups);
    }
    else
#endif
    {
        custom_conv_ups_zxz1_cpu(ks/2, weightsx4xpose, h_in_padded, w_in_padded, in_padded, padded_ups);
    }

    int new_w_conv = (w_in_padded-(ks/2)+1)*2;

    // want to remove upsampling crop
    int new_crop = ks/2+1;

    // emit the desired dimsions, skipping initial crop.
    for (int y = 0; y < h_target; y++)
        memcpy(out+y*stride_target, padded_ups+(y+new_crop)*new_w_conv+new_crop, w_target*sizeof(out[0]));
}

// dequant int weights
void arm_scale(int32_t *wb, int n_wb, int q_step_shift, bool bias)
{
    // dequant & scale back to int.
    for (int i = 0; i < n_wb; i++)
    {
        // A q_step of Q implies a quant of 1/(1<<Q)
        // eg 0 maps to 1
        // eg 6 maps to 1/64
        // We take this to a fixed point precision of 6 with a single shift left.
        int delta; // we want to shift left by this if it is positive.
        if (bias)
            delta = FPSHIFT*2 - q_step_shift;
        else
            delta = FPSHIFT - q_step_shift;
        if (delta > 0)
            if (wb[i] < 0)
                wb[i] = -(int)((-wb[i])<<delta);
            else
                wb[i] = (int)((wb[i])<<delta);
    }
}

int32_t *decode_weights(TDecBinCABAC *cabac, int count, int n_weights)
{
    int32_t *result = (int32_t *)aligned_alloc(ALIGNTO, ALIGN(n_weights*sizeof(result[0])));
    if (result == NULL)
    {
        printf("cannot aligned_alloc %d weights\n", n_weights);
        exit(1);
    }
    for (int i = 0; i < n_weights; i++)
    {
        int val = cabac->decodeExGolomb(count);
        if (val != 0)
        {
            if (cabac->decodeBinEP() != 0)
                val = -val;
        }
        result[i] = val;
    }
    return result;
}

// decode with dequant to float
float *decode_weights_q(TDecBinCABAC *cabac, int count, int n_weights, int q_step_shift)
{
    float *result = (float *)aligned_alloc(ALIGNTO, ALIGN(n_weights*sizeof(result[0])));
    if (result == NULL)
    {
        printf("cannot aligned_alloc %d weights\n", n_weights);
        exit(1);
    }
    for (int i = 0; i < n_weights; i++)
    {
        int tmp = cabac->decodeExGolomb(count);
        if (tmp != 0)
        {
            if (cabac->decodeBinEP() != 0)
                tmp = -tmp;
        }
        result[i] = tmp*1.0/(1<<q_step_shift);
    }
    return result;
}

struct frame_memory
{
    float *raw_frame;
    int    origin_idx;
    int    h;
    int    w;
    int    stride;
    int    plane_stride;
};

struct frame_memory decode_frame(struct cc_bs_gop_header &gop_header, struct cc_bs_frame *frame_coded)
{
    struct cc_bs_frame_header &frame_header = frame_coded->m_frame_header;
    int32_t *mlpw[frame_header.n_hidden_layers_arm];
    int32_t *mlpw_t[frame_header.n_hidden_layers_arm];
    int32_t *mlpb[frame_header.n_hidden_layers_arm];
    int32_t *mlpwOUT = NULL;
    int32_t *mlpbOUT = NULL;

    float *upsw = NULL;

    float *synw[frame_header.n_syn_layers];
    float *synb[frame_header.n_syn_layers];

    // READ ARM
    {
        InputBitstream bs_weights;
        std::vector<UChar> &bs_fifo_weights = bs_weights.getFifo();
        InputBitstream bs_biases;
        std::vector<UChar> &bs_fifo_biases = bs_biases.getFifo();

        TDecBinCABAC cabac_weights;
        TDecBinCABAC cabac_biases;

        bs_fifo_weights = frame_coded->m_arm_weights_hevc;
        cabac_weights.init(&bs_weights);
        cabac_weights.start();

        bs_fifo_biases = frame_coded->m_arm_biases_hevc;
        cabac_biases.init(&bs_biases);
        cabac_biases.start();

        struct cc_bs_layer_quant_info &arm_lqi = frame_header.arm_lqi;

        //auto weights_ctxs = &g_contexts[ZERO_MU][arm_lqi.scale_index_nn_weight];
        //auto biases_ctxs = &g_contexts[ZERO_MU][arm_lqi.scale_index_nn_bias];

        int q_step_w_shift = Q_STEP_ARM_WEIGHT_SHIFT[arm_lqi.q_step_index_nn_weight];
        int q_step_b_shift = Q_STEP_ARM_BIAS_SHIFT[arm_lqi.q_step_index_nn_bias];

        for (int idx = 0; idx < frame_header.n_hidden_layers_arm; idx++)
        {
            int n_weights = frame_header.dim_arm*frame_header.dim_arm;
            int n_biases = frame_header.dim_arm;
            mlpw[idx] = decode_weights(&cabac_weights, arm_lqi.scale_index_nn_weight, n_weights);
            arm_scale(mlpw[idx], n_weights, q_step_w_shift, false);
            mlpb[idx] = decode_weights(&cabac_biases,  arm_lqi.scale_index_nn_bias, n_biases);
            arm_scale(mlpb[idx], n_biases, q_step_b_shift, true);
        }
        mlpwOUT = decode_weights(&cabac_weights, arm_lqi.scale_index_nn_weight, frame_header.dim_arm*2);
        arm_scale(mlpwOUT, frame_header.dim_arm*2, q_step_w_shift, false);
        mlpbOUT = decode_weights(&cabac_biases,  arm_lqi.scale_index_nn_bias, 2);
        arm_scale(mlpbOUT, 2, q_step_b_shift, true);

        /* transpose arm weights from row=in to row=out */
        int arm_n_features = frame_header.dim_arm;

        for (int hl = 0; hl < frame_header.n_hidden_layers_arm; hl++)
        {
            mlpw_t[hl] = (int32_t *)aligned_alloc(ALIGNTO, ALIGN(arm_n_features*arm_n_features*sizeof(mlpw_t[hl][0])));
            if (mlpw_t[hl] == NULL)
            {
                printf("cannot aligned_alloc %d arm features\n", arm_n_features*arm_n_features);
                exit(1);
            }
            int kidx = 0;
            for (int ky = 0; ky < arm_n_features; ky++)
                for (int kx = 0; kx < arm_n_features; kx++, kidx++)
                {
                    mlpw_t[hl][kidx] = mlpw[hl][kx*arm_n_features+ky];
                    mlpw_t[hl][kidx] = mlpw[hl][kx*arm_n_features+ky];
                }
        }
    }

    // READ UPS
    {
        InputBitstream bs_weights;
        std::vector<UChar> &bs_fifo_weights = bs_weights.getFifo();

        TDecBinCABAC cabac_weights;

        bs_fifo_weights = frame_coded->m_ups_weights_hevc;
        cabac_weights.init(&bs_weights);
        cabac_weights.start();
        struct cc_bs_layer_quant_info &ups_lqi = frame_header.ups_lqi;

        //auto weights_ctxs = &g_contexts[ZERO_MU][ups_lqi.scale_index_nn_weight];
        int q_step_w_shift = Q_STEP_UPS_SHIFT[ups_lqi.q_step_index_nn_weight];

        upsw = decode_weights_q(&cabac_weights, ups_lqi.scale_index_nn_weight, frame_header.upsampling_kern_size*frame_header.upsampling_kern_size, q_step_w_shift);

        if (frame_header.static_upsampling_kernel != 0)
        {

            float bilinear_kernel[4 * 4] = {
                0.0625, 0.1875, 0.1875, 0.0625,
                0.1875, 0.5625, 0.5625, 0.1875,
                0.1875, 0.5625, 0.5625, 0.1875,
                0.0625, 0.1875, 0.1875, 0.0625
            };

            float bicubic_kernel[8 * 8] = {
                 0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619,
                 0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857,
                -0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498,
                -0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479,
                -0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479,
                -0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498,
                 0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857,
                 0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619
            };

            // Use bilinear static kernel if kernel size is smaller than 8, else bicubic
            int static_kernel_size = frame_header.upsampling_kern_size < 8 ? 4 : 8;
            float * static_kernel = frame_header.upsampling_kern_size < 8 ? (float *) bilinear_kernel : (float *) bicubic_kernel;

            // First: re-init everything with zeros
            for (int r = 0; r < frame_header.upsampling_kern_size; r++)
            {
                for (int c = 0; c < frame_header.upsampling_kern_size; c++)
                {
                    upsw[r * frame_header.upsampling_kern_size + c] = 0.0;
                }
            }

            // Then, write the values of the static filter
            // Top left coordinate of the first filter coefficient
            int top_left_coord = (frame_header.upsampling_kern_size - static_kernel_size) / 2;
            for (int r = 0; r < static_kernel_size; r++)
            {
                for (int c = 0; c < static_kernel_size; c++)
                {
                    upsw[(r + top_left_coord) * frame_header.upsampling_kern_size + c  + top_left_coord] = static_kernel[r * static_kernel_size + c];
                }
            }
        }
    }

    // READ SYN
    {
        InputBitstream bs_weights;
        std::vector<UChar> &bs_fifo_weights = bs_weights.getFifo();
        InputBitstream bs_biases;
        std::vector<UChar> &bs_fifo_biases = bs_biases.getFifo();

        TDecBinCABAC cabac_weights;
        TDecBinCABAC cabac_biases;

        bs_fifo_weights = frame_coded->m_syn_weights_hevc;
        bs_fifo_biases = frame_coded->m_syn_biases_hevc;
        cabac_weights.init(&bs_weights);
        cabac_weights.start();
        cabac_biases.init(&bs_biases);
        cabac_biases.start();
        struct cc_bs_layer_quant_info &syn_lqi = frame_header.syn_lqi;

        //auto weights_ctxs = &g_contexts[ZERO_MU][syn_lqi.scale_index_nn_weight];
        //auto biases_ctxs = &g_contexts[ZERO_MU][syn_lqi.scale_index_nn_bias];

        int q_step_w_shift = Q_STEP_SYN_SHIFT[syn_lqi.q_step_index_nn_weight];
        int q_step_b_shift = Q_STEP_SYN_SHIFT[syn_lqi.q_step_index_nn_bias];

        int n_in_ft = frame_header.n_latent_n_resolutions; // !!! features per layer
        for (int idx = 0; idx < frame_header.n_syn_layers; idx++)
        {
            struct cc_bs_syn_layer &syn = frame_header.layers_synthesis[idx];
            int n_weights = n_in_ft * syn.ks*syn.ks * syn.n_out_ft;
            int n_biases = syn.n_out_ft;

            synw[idx] = decode_weights_q(&cabac_weights, syn_lqi.scale_index_nn_weight, n_weights, q_step_w_shift);
            synb[idx] = decode_weights_q(&cabac_biases,  syn_lqi.scale_index_nn_bias, n_biases,  q_step_b_shift);

            n_in_ft = syn.n_out_ft;
        }
    }

    printf("loaded weights\n");
    fflush(stdout);

    // we allocate a max-sized (ignoring padding) buffer, suitable for holding
    // upsample final output and synthesis outputs during syn processing.

    int const latent_n_resolutions = frame_header.n_latent_n_resolutions;
    int const n_ctx_rowcol = 4; // !!! temporary semi-wave transition g_info.n_ctx_rowcol;
    int const hls_sig_blksize = frame_header.hls_sig_blksize;

    int n_max_planes = latent_n_resolutions;
    int n_max_pad = 0; // during synthesis.
    for (int syn_idx = 0; syn_idx < frame_header.n_syn_layers; syn_idx++)
    {
        int n_syn_out = frame_header.layers_synthesis[syn_idx].n_out_ft;
        if (n_syn_out > n_max_planes)
            n_max_planes = n_syn_out;
        int n_pad = frame_header.layers_synthesis[syn_idx].ks/2;
        if (n_pad > n_max_pad)
            n_max_pad = n_pad;
    }
    printf("MAX PLANES SET TO %d; MAX PAD SET TO %d\n", n_max_planes, n_max_pad);

    int const ups_stride = gop_header.img_w+2*n_max_pad;
    int const ups_plane_stride = ups_stride*(gop_header.img_h+2*n_max_pad);
    int const ups_origin_idx = n_max_pad*ups_stride+n_max_pad;
    float *ups_by_hand = new float[ups_plane_stride*n_max_planes];
    // allocate secondary ups buffer if synthesis cannot be done in place.
    float *ups_by_hand2 = n_max_pad == 0 ? NULL : new float[ups_plane_stride*n_max_planes];

    float *ups_by_hand_layers[latent_n_resolutions];
    ups_by_hand_layers[0] = ups_by_hand;
    for (int i = 1; i < latent_n_resolutions; i++)
        ups_by_hand_layers[i] = ups_by_hand_layers[i-1]+(gop_header.img_h+2*n_max_pad)*ups_stride;

    // RUN ARM
    std::vector<float *> decoded_layers_raw; // NULL implies empty; the first (ie, fullres) is targeting ups_by_hand_layers[origidx]

    int pad = n_ctx_rowcol;       // by how much the frame is padded.

    // full res Y for latents.
    int *full_res_y =  new int[(gop_header.img_w+2*pad)*(gop_header.img_h+2*pad)];

    // latent-layer processing.
    for (int layer_number = 0, h_grid = gop_header.img_h, w_grid = gop_header.img_w;
         layer_number < latent_n_resolutions;
         layer_number++, h_grid = (h_grid+1)/2, w_grid = (w_grid+1)/2)
    {
        int const Y_stride = w_grid+2*pad;

        printf("starting layer %d\n", layer_number);
        fflush(stdout);

        auto &bytes = frame_coded->m_latents_hevc[layer_number];
        if (bytes.size() == 0)
        {
            // produce a zero layer.
            decoded_layers_raw.push_back(NULL);
            continue;
        }

        memset(full_res_y, 0, Y_stride*(h_grid+pad+pad)*sizeof(full_res_y[0])); // !!! just padding

        // BAC decoding:
        InputBitstream bsBAC;
        std::vector<UChar> &bs = bsBAC.getFifo();
        bs = bytes;

        TDecBinCABAC layerBAC;
        layerBAC.init(&bsBAC);
        layerBAC.start();

        BACContext bac_context;
        bac_context.set_layer(&layerBAC, h_grid, w_grid, hls_sig_blksize);

        const auto time_arm_start = std::chrono::steady_clock::now();

#ifdef CCDECAPI_AVX2
        if (frame_header.dim_arm == 8)
            custom_conv_11_int32_avx2_8_X_X(
                                      mlpw_t, mlpb,
                                      mlpwOUT, mlpbOUT,
                                      get_context_indicies(frame_header.dim_arm, pad+w_grid+pad), frame_header.dim_arm, frame_header.n_hidden_layers_arm,
                                      &full_res_y[pad*(pad+w_grid+pad)+pad],
                                      h_grid, w_grid, pad,
                                      bac_context
                                      );
        else if (frame_header.dim_arm == 16)
            custom_conv_11_int32_avx2_16_X_X(
                                      mlpw_t, mlpb,
                                      mlpwOUT, mlpbOUT,
                                      get_context_indicies(frame_header.dim_arm, pad+w_grid+pad), frame_header.dim_arm, frame_header.n_hidden_layers_arm,
                                      &full_res_y[pad*(pad+w_grid+pad)+pad],
                                      h_grid, w_grid, pad,
                                      bac_context
                                      );
        else if (frame_header.dim_arm == 24)
            custom_conv_11_int32_avx2_24_X_X(
                                      mlpw_t, mlpb,
                                      mlpwOUT, mlpbOUT,
                                      get_context_indicies(frame_header.dim_arm, pad+w_grid+pad), frame_header.dim_arm, frame_header.n_hidden_layers_arm,
                                      &full_res_y[pad*(pad+w_grid+pad)+pad],
                                      h_grid, w_grid, pad,
                                      bac_context
                                      );
        else if (frame_header.dim_arm == 32)
            custom_conv_11_int32_avx2_32_X_X(
                                      mlpw_t, mlpb,
                                      mlpwOUT, mlpbOUT,
                                      get_context_indicies(frame_header.dim_arm, pad+w_grid+pad), frame_header.dim_arm, frame_header.n_hidden_layers_arm,
                                      &full_res_y[pad*(pad+w_grid+pad)+pad],
                                      h_grid, w_grid, pad,
                                      bac_context
                                      );
        else
#endif
            custom_conv_11_int32_cpu_X_X_X(
                                      mlpw_t, mlpb,
                                      mlpwOUT, mlpbOUT,
                                      get_context_indicies(frame_header.dim_arm, pad+w_grid+pad), frame_header.dim_arm, frame_header.n_hidden_layers_arm,

                                      &full_res_y[pad*(pad+w_grid+pad)+pad],
                                      h_grid, w_grid, pad,

                                      bac_context
                                      );

        const auto time_arm_done = std::chrono::steady_clock::now();
        const std::chrono::duration<double> arm_elapsed = (time_arm_done-time_arm_start);
        time_arm_seconds += (float)arm_elapsed.count();

        // float the result.
        float *raw_layer = layer_number == 0 ? &ups_by_hand_layers[0][ups_origin_idx] : new float[h_grid*w_grid];
        int raw_pad = layer_number == 0 ? n_max_pad : 0;
        for (int y = 0, srcidx = pad*Y_stride+pad, dstidx = 0; y < h_grid; y++, srcidx += pad+pad, dstidx += raw_pad+raw_pad)
        {
            for (int x = 0; x < w_grid; x++, srcidx++, dstidx++)
            {
                raw_layer[dstidx] = (float)(full_res_y[srcidx]/FPSCALE);
            }
        }
        decoded_layers_raw.push_back(raw_layer);

    } // layer.

    delete [] full_res_y;
    full_res_y = NULL;
    printf("arm done!\n");

    // NEW UPSAMPLE
    const auto time_ups_start = std::chrono::steady_clock::now();

    float *tmp0 = (float *)new float[gop_header.img_h*gop_header.img_w];
    float *tmp1 = (float *)new float[gop_header.img_h*gop_header.img_w];
    float *tmps[2] = {tmp0, tmp1};
    int ups_ks = frame_header.upsampling_kern_size;
    float upsxpose[ups_ks][ups_ks];  // conv transpose.
    for (int y = 0; y < ups_ks; y++)
    {
        for (int x = 0; x < ups_ks; x++)
        {
            upsxpose[y][x] = upsw[(ups_ks-1-y)*ups_ks+(ups_ks-1-x)];
        }
    }

    // extract 4 smaller filters. 8x8 -> 4x 4x4
    // f0 operates at origin, y=0, x=0
    // f1 operates at y=0, x=1
    // f2 operates at y=1, x=0
    // f3 operates at y=1, x=1
    float weightsx4xpose[4][ups_ks/2][ups_ks/2]; // [filterno][h][w]
    for (int f = 0; f < 4; f++)
    {
        int fbase_y = 1-f/2;
        int fbase_x = 1-f%2;
        for (int y = 0; y < ups_ks/2; y++)
        {
            for (int x = 0; x < ups_ks/2; x++)
            {
                weightsx4xpose[f][y][x] = upsxpose[fbase_y+2*y][fbase_x+2*x];
            }
        }
    }

    int h_pyramid[latent_n_resolutions];
    int w_pyramid[latent_n_resolutions];

    for (int layer_number = 0, h_grid = gop_header.img_h, w_grid = gop_header.img_w;
         layer_number < latent_n_resolutions;
         layer_number++, h_grid = (h_grid+1)/2, w_grid = (w_grid+1)/2)
    {
        h_pyramid[layer_number] = h_grid;
        w_pyramid[layer_number] = w_grid;
    }

    for (int layer_number = 0, h_grid = gop_header.img_h, w_grid = gop_header.img_w;
         layer_number < latent_n_resolutions;
         layer_number++, h_grid = (h_grid+1)/2, w_grid = (w_grid+1)/2)
    {
        if (decoded_layers_raw[layer_number] == NULL)
        {
            // no need to upsample.  just zero the content (including any full-res layer).
            memset(ups_by_hand_layers[layer_number], 0, (h_grid+2*n_max_pad)*ups_stride*sizeof(ups_by_hand_layers[0][0]));
            continue;
        }
        if (layer_number == 0)
        {
            // full res. if non-null, already present in ups_by_hand_layers[0].
            continue;
        }
        // continually scale up to the final resolution.
        // the final resolution arrives directly into ups_by_hand_layers[x]
        float *lo_res = decoded_layers_raw[layer_number];
        int tmptarget = 0;
        for (int target_layer = layer_number-1; target_layer >= 0; target_layer--)
        {
            int h_target = h_pyramid[target_layer];
            int w_target = w_pyramid[target_layer];

            float *out;
            int out_stride;
            if (target_layer == 0)
            {
                out = &ups_by_hand_layers[layer_number][ups_origin_idx];
                out_stride = ups_stride;
            }
            else
            {
                out = tmps[tmptarget];
                out_stride = w_target;
            }
            custom_upsample_4(ups_ks, &weightsx4xpose[0][0][0], h_pyramid[target_layer+1], w_pyramid[target_layer+1],
                              lo_res, out, h_target, w_target, out_stride);
            lo_res = tmps[tmptarget];
            tmptarget = 1-tmptarget;
        }
        delete [] decoded_layers_raw[layer_number];
    }
    delete tmps[0];
    delete tmps[1];

    const auto time_ups_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> hand_ups_elapsed_seconds = time_ups_end - time_ups_start;
    time_ups_seconds += hand_ups_elapsed_seconds.count();

    // NEW SYNTHESIS: IN-PLACE CONV
    int    n_syn_in = latent_n_resolutions;
    float *syn_in = ups_by_hand;
    float *syn_out = ups_by_hand2; // if cannot do syn in-place.
    const auto time_syn_start = std::chrono::steady_clock::now();

    for (int syn_idx = 0; syn_idx < frame_header.n_syn_layers; syn_idx++)
    {
        int n_syn_out = frame_header.layers_synthesis[syn_idx].n_out_ft;
        printf("SYN%d: #in=%d #out=%d\n", syn_idx, n_syn_in, n_syn_out);
        if (frame_header.layers_synthesis[syn_idx].ks > 1)
        {
            // pad syn_in, and direct output to syn_out.
            pad = frame_header.layers_synthesis[syn_idx].ks/2;
            int w_padded = gop_header.img_w+2*pad;
            int h_padded = gop_header.img_h+2*pad;
            int pad_origin_idx = ups_origin_idx-pad*ups_stride-pad;
            for (int l = 0; l < n_syn_in; l++)
                custom_pad_replicate_plane_in_place(gop_header.img_h, gop_header.img_w, ups_stride, &syn_in[l*ups_plane_stride+ups_origin_idx], pad);
            printf("... padded\n");

#ifdef CCDECAPI_AVX2
            if (frame_header.layers_synthesis[syn_idx].ks == 3 && n_syn_in == 3 && n_syn_out == 3)
                custom_conv_ks3_in3_out3_avx2(frame_header.layers_synthesis[syn_idx].ks, synw[syn_idx], synb[syn_idx],
                                                h_padded, w_padded, ups_stride, ups_plane_stride, ups_origin_idx-pad_origin_idx, n_syn_in, &syn_in[pad_origin_idx], n_syn_out, &syn_out[ups_origin_idx],
                                                !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else if (frame_header.layers_synthesis[syn_idx].ks == 3 && n_syn_in == 9 && n_syn_out == 6)
                custom_conv_ks3_in9_out6_avx2(frame_header.layers_synthesis[syn_idx].ks, synw[syn_idx], synb[syn_idx],
                                                h_padded, w_padded, ups_stride, ups_plane_stride, ups_origin_idx-pad_origin_idx, n_syn_in, &syn_in[pad_origin_idx], n_syn_out, &syn_out[ups_origin_idx],
                                                !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else
                custom_conv_ksX_inX_outX_avx2(frame_header.layers_synthesis[syn_idx].ks, synw[syn_idx], synb[syn_idx],
                                                h_padded, w_padded, ups_stride, ups_plane_stride, ups_origin_idx-pad_origin_idx, n_syn_in, &syn_in[pad_origin_idx], n_syn_out, &syn_out[ups_origin_idx],
                                                !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
#else
            custom_conv_ksX_inX_outX(frame_header.layers_synthesis[syn_idx].ks, synw[syn_idx], synb[syn_idx],
                                            h_padded, w_padded, ups_stride, ups_plane_stride, ups_origin_idx-pad_origin_idx, n_syn_in, &syn_in[pad_origin_idx], n_syn_out, &syn_out[ups_origin_idx],
                                            !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
#endif

            // swap our in & out.
            float *tmp = syn_in;
            syn_in = syn_out;
            syn_out = tmp;
        }
        else
        {
#ifdef CCDECAPI_AVX2
            if (n_syn_in == 7 && n_syn_out == 9)
                custom_conv_ks1_in7_out9_avx2(1, synw[syn_idx], synb[syn_idx], gop_header.img_h, gop_header.img_w, ups_stride, ups_plane_stride, 0, n_syn_in, &syn_in[ups_origin_idx], n_syn_out, &syn_in[ups_origin_idx], !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else if (n_syn_in == 9 && n_syn_out == 3)
                custom_conv_ks1_in9_out3_avx2(1, synw[syn_idx], synb[syn_idx], gop_header.img_h, gop_header.img_w, ups_stride, ups_plane_stride, 0, n_syn_in, &syn_in[ups_origin_idx], n_syn_out, &syn_in[ups_origin_idx], !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else if (n_syn_in == 9 && n_syn_out == 6)
                custom_conv_ks1_in9_out6_avx2(1, synw[syn_idx], synb[syn_idx], gop_header.img_h, gop_header.img_w, ups_stride, ups_plane_stride, 0, n_syn_in, &syn_in[ups_origin_idx], n_syn_out, &syn_in[ups_origin_idx], !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else if (n_syn_in == 9 && n_syn_out == 9)
                custom_conv_ks1_in9_out9_avx2(1, synw[syn_idx], synb[syn_idx], gop_header.img_h, gop_header.img_w, ups_stride, ups_plane_stride, 0, n_syn_in, &syn_in[ups_origin_idx], n_syn_out, &syn_in[ups_origin_idx], !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else
                custom_conv_ks1_inX_outX_avx2(1, synw[syn_idx], synb[syn_idx], gop_header.img_h, gop_header.img_w, ups_stride, ups_plane_stride, 0, n_syn_in, &syn_in[ups_origin_idx], n_syn_out, &syn_in[ups_origin_idx], !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
#else
            custom_conv_ks1_inX_outX(1, synw[syn_idx], synb[syn_idx], gop_header.img_h, gop_header.img_w, ups_stride, ups_plane_stride, 0, n_syn_in, &syn_in[ups_origin_idx], n_syn_out, &syn_in[ups_origin_idx], !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);

#endif
        }

        n_syn_in = n_syn_out;
    }

    const auto time_syn_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> syn_elapsed_seconds = time_syn_end - time_syn_start;
    time_syn_seconds += syn_elapsed_seconds.count();

    for (int hl = 0; hl < frame_header.n_hidden_layers_arm; hl++)
    {
        free(mlpw[hl]);
        free(mlpb[hl]);
        free(mlpw_t[hl]);
    }
    free(mlpwOUT);
    free(mlpbOUT);

    // !!! possibly more freeing to do
    for (int i = 0; i < frame_header.n_syn_layers; i++)
    {
        free(synw[i]);
        free(synb[i]);
    }
    free(upsw);

    struct frame_memory result;
    result.raw_frame = syn_in;
    result.origin_idx = ups_origin_idx;
    result.h = gop_header.img_h;
    result.w = gop_header.img_w;
    result.stride = ups_stride;
    result.plane_stride = ups_plane_stride;

    if (syn_in != ups_by_hand)
    {
        delete[] ups_by_hand;
    }
    else if (ups_by_hand2 != NULL)
    {
        delete[] ups_by_hand2;
    }

    return result; // raw frame either [3] (intra) or [6 or 9] (inter)
}

// only use nc == 3
void ppm_out(int nc, int pixel_depth, struct frame_memory &in_info, char const *outname)
{
    int const h = in_info.h;
    int const w = in_info.w;
    int const stride = in_info.stride;
    int const plane_stride = in_info.plane_stride;
    float *ins = &in_info.raw_frame[in_info.origin_idx];
    FILE *fout = fopen(outname, "wb");
    if (fout == NULL)
    {
        printf("Cannot open %s for writing.\n", outname);
        exit(1);
    }

    if (nc == 3)
    {
        fprintf(fout, "P6\n");
        fprintf(fout, "%d %d\n", w, h);
        fprintf(fout, "255\n");
        float *inR = ins;
        float *inG = inR+plane_stride;
        float *inB = inG+plane_stride;
        for (int y = 0; y < h; y++, inR += stride-w, inG += stride-w, inB += stride-w)
            for (int x = 0; x < w; x++)
            {
                int r = (*inR++)*255+0.5;
                int g = (*inG++)*255+0.5;
                int b = (*inB++)*255+0.5;
                if (r < 0) r = 0;
                if (g < 0) g = 0;
                if (b < 0) b = 0;
                if (r > 255) r = 255;
                if (g > 255) g = 255;
                if (b > 255) b = 255;
                unsigned char pix[3];
                pix[0] = r;
                pix[1] = g;
                pix[2] = b;
                fwrite(pix, 3, 1, fout);
            }
    }

    fclose(fout);
}

// we 'stabilise' the yuv to (0..255) as integers, as well as uv subsample.
// incoming 444 is planar, outgoing 420 is planar.
unsigned char *convert_444_420_8b(struct frame_memory in)
{
    int const h = in.h;
    int const w = in.w;
    int const in_stride = in.stride;
    unsigned char *out = new unsigned char[h*w*3/2];
    float *src = &in.raw_frame[in.origin_idx];
    unsigned char *dst = out;

    // Y.
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++, dst++)
        {
            int v255 = *src*255+0.5;
            if (v255 < 0)
                v255 = 0;
            else if (v255 > 255)
                v255 = 255;
            *dst = v255;
        }

    // U. 'nearest'
    src = &in.raw_frame[1*in.plane_stride+in.origin_idx];
    for (int y = 0; y < h/2; y++, src += in_stride-w+in_stride)
        for (int x = 0; x < w/2; x++, src += 2, dst++)
        {
            int v255 = *src*255+0.5;
            if (v255 < 0)
                v255 = 0;
            else if (v255 > 255)
                v255 = 255;
            *dst = v255;
        }

    // V. 'nearest'
    src = &in.raw_frame[2*in.plane_stride+in.origin_idx];
    for (int y = 0; y < h/2; y++, src += in_stride-w+in_stride)
        for (int x = 0; x < w/2; x++, src += 2, dst++)
        {
            int v255 = *src*255+0.5;
            if (v255 < 0)
                v255 = 0;
            else if (v255 > 255)
                v255 = 255;
            *dst = v255;
        }

    return out;
}

// we 'stabilise' the yuv to (0..1023) as integers, as well as uv subsample.
// incoming 444 is planar, outgoing 420 is planar.
unsigned short *convert_444_420_10b(struct frame_memory in)
{
    int const h = in.h;
    int const w = in.w;
    int const in_stride = in.stride;
    unsigned short *out = new unsigned short[h*w*3/2];
    float *src = &in.raw_frame[in.origin_idx];
    unsigned short *dst = out;

    // Y.
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++, dst++)
        {
            int v1023 = *src*1023+0.5;
            if (v1023 < 0)
                v1023 = 0;
            else if (v1023 > 1023)
                v1023 = 1023;
            *dst = v1023;
        }

    // U. 'nearest'
    src = &in.raw_frame[1*in.plane_stride+in.origin_idx];
    for (int y = 0; y < h/2; y++, src += in_stride-w+in_stride)
        for (int x = 0; x < w/2; x++, src += 2, dst++)
        {
            int v1023 = *src*1023+0.5;
            if (v1023 < 0)
                v1023 = 0;
            else if (v1023 > 1023)
                v1023 = 1023;
            *dst = v1023;
        }

    // V. 'nearest'
    src = &in.raw_frame[2*in.plane_stride+in.origin_idx];
    for (int y = 0; y < h/2; y++, src += in_stride-w+in_stride)
        for (int x = 0; x < w/2; x++, src += 2, dst++)
        {
            int v1023 = *src*1023+0.5;
            if (v1023 < 0)
                v1023 = 0;
            else if (v1023 > 1023)
                v1023 = 1023;
            *dst = v1023;
        }

    return out;
}


// incoming 420 is planar, outgoing 444 is planar.
struct frame_memory convert_420_444_8b(unsigned char *in, int h, int w)
{
    float *out = new float[3*h*w];

    unsigned char *src = in;
    float *dst = out;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            *dst = *src/255.0;

    // two output lines per input.
    float *dst0 = dst;
    float *dst1 = dst+w;
    for (int y = 0; y < h/2; y++, dst0 += w, dst1 += w)
        for (int x = 0; x < w/2; x++, src++)
        {
            float v = *src/255.0;
            *dst0++ = v;
            *dst0++ = v;
            *dst1++ = v;
            *dst1++ = v;
        }

    for (int y = 0; y < h/2; y++, dst0 += w, dst1 += w)
        for (int x = 0; x < w/2; x++, src++)
        {
            float v = *src/255.0;
            *dst0++ = v;
            *dst0++ = v;
            *dst1++ = v;
            *dst1++ = v;
        }

    struct frame_memory result;
    result.raw_frame = out;
    result.origin_idx = 0;
    result.h = h;
    result.w = w;
    result.stride = w;
    result.plane_stride = h*w;

    return result;
}

// incoming 420 is planar, outgoing 444 is planar.
struct frame_memory convert_420_444_10b(unsigned short *in, int h, int w)
{
    float *out = new float[3*h*w];

    unsigned short *src = in;
    float *dst = out;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            *dst = *src/1023.0;

    // two output lines per input.
    float *dst0 = dst;
    float *dst1 = dst+w;
    for (int y = 0; y < h/2; y++, dst0 += w, dst1 += w)
        for (int x = 0; x < w/2; x++, src++)
        {
            float v = *src/1023.0;
            *dst0++ = v;
            *dst0++ = v;
            *dst1++ = v;
            *dst1++ = v;
        }

    for (int y = 0; y < h/2; y++, dst0 += w, dst1 += w)
        for (int x = 0; x < w/2; x++, src++)
        {
            float v = *src/1023.0;
            *dst0++ = v;
            *dst0++ = v;
            *dst1++ = v;
            *dst1++ = v;
        }

    struct frame_memory result;
    result.raw_frame = out;
    result.origin_idx = 0;
    result.h = h;
    result.w = w;
    result.stride = w;
    result.plane_stride = h*w;

    return result;
}




// frame_420: byte, planar, seeks to frame position.
void dump_yuv420_8b(unsigned char *frame_420, int h, int w, FILE *fout, int frame_idx)
{
    fseek(fout, frame_idx*h*w*3/2, SEEK_SET);
    fwrite(frame_420, sizeof(frame_420[0]), h*w*3/2, fout);
    fflush(fout);
}

// frame_420: byte, planar, seeks to frame position.
void dump_yuv420_10b(unsigned short *frame_420, int h, int w, FILE *fout, int frame_idx)
{
    fseek(fout, frame_idx*h*w*3/2, SEEK_SET);
    fwrite(frame_420, sizeof(frame_420[0]), h*w*3/2, fout);
    fflush(fout);
}


// frame_444: float, planar, transformed to byte 444 planar output.
void dump_yuv444(float *frame_444, int h, int w, int pixel_depth, std::string outyuv)
{
    // incoming frame is planar.
    unsigned char *f = new unsigned char[h*w*3];
    unsigned char *y = f+w*h*0;
    unsigned char *u = f+w*h*1;
    unsigned char *v = f+w*h*2;
    for (int p = 0; p < w*h*pixel_depth; p += pixel_depth)
    {
        int val = (int)(frame_444[p+0]*255+0.5);
        if (val < 0)
            val = 0;
        else if (val > 255)
            val = 255;
        *y++ = val;
        val = (int)(frame_444[p+1]*255+0.5);
        if (val < 0)
            val = 0;
        else if (val > 255)
            val = 255;
        *u++ = val;
        val = (int)(frame_444[p+2]*255+0.5);
        if (val < 0)
            val = 0;
        else if (val > 255)
            val = 255;
        *v++ = val;
    }

    FILE *fout = fopen(outyuv.c_str(), "wb");
    if (fout == NULL)
    {
        printf("Cannot open %s for writing\n", outyuv.c_str());
        exit(1);
    }
    fwrite(f, sizeof(f[0]), h*w*3, fout);
    fclose(fout);
    printf("wrote %s, size h=%d w=%d\n", outyuv.c_str(), h, w);
    fflush(stdout);
    delete[] f;
}

// returns a yuv444 float, warping a reference, apply a gain
// bilinear sampling.
// ref: float, planar.
struct frame_memory warp(struct frame_memory &raw_info, int raw_info_depth, struct frame_memory &ref, int raw_xyidx, int raw_gainidx, int flo_gain, bool add_residue)
{
    const auto time_warp_start = std::chrono::steady_clock::now();

    float *src = &ref.raw_frame[ref.origin_idx];
    int const src_stride = ref.stride;
    int const src_plane_stride = ref.plane_stride;
    float *raw = &raw_info.raw_frame[raw_info.origin_idx];
    int const raw_stride = raw_info.stride;
    int const raw_plane_stride = raw_info.plane_stride;
    int const h = raw_info.h;
    int const w = raw_info.w;

    float *out = new float[3*h*w]; // yuv444 = warp(ref)*gain + residue, planar.
    int const out_plane_stride = h*w;

    float *dst = out; // scans through first plane.

    for (int y = 0; y < h; y++, raw += raw_stride-w)
    for (int x = 0; x < w; x++, raw++, dst++)
    {
        float px = (raw[(raw_xyidx+0)*raw_plane_stride]*flo_gain)+x;
        float py = (raw[(raw_xyidx+1)*raw_plane_stride]*flo_gain)+y;

        int basex0 = (int)(floor(px)); // integer coord 'left'
        int basex1 = basex0+1;
        float dx = px-basex0;
        if (basex0 < 0)
        {
            basex0 = 0;
            basex1 = basex0;
            dx = 0;
        }
        else if (basex0 >= w-1)
        {
            basex0 = w-1;
            basex1 = basex0;
            dx = 0;
        }
        int basey0 = (int)(floor(py));
        int basey1 = basey0+1;
        float dy = py-(basey0);
        if (basey0 < 0)
        {
            basey0 = 0;
            basey1 = basey0;
            dy = 0;
        }
        else if (basey0 >= h-1)
        {
            basey0 = h-1;
            basey1 = basey0;
            dy = 0;
        }

        float gain;
        if (raw_gainidx < 0)
            gain = raw[-raw_gainidx*raw_plane_stride]+0.5;
        else
            gain = raw[raw_gainidx*raw_plane_stride]+0.5;
        if (gain < 0.0)
            gain = 0.0;
        else if (gain > 1.0)
            gain = 1.0;
        if (raw_gainidx < 0)
            gain = 1-gain;

        float *A = &src[((basey0)*src_stride+basex0)];
        float *B = &src[((basey0)*src_stride+basex1)];
        float *C = &src[((basey1)*src_stride+basex0)];
        float *D = &src[((basey1)*src_stride+basex1)];

        float h0 = A[0*src_plane_stride]+(B[0*src_plane_stride]-A[0*src_plane_stride])*dx;
        float h1 = C[0*src_plane_stride]+(D[0*src_plane_stride]-C[0*src_plane_stride])*dx;
        dst[0*out_plane_stride] = (h0+(h1-h0)*dy)*gain;

        h0 = A[1*src_plane_stride]+(B[1*src_plane_stride]-A[1*src_plane_stride])*dx;
        h1 = C[1*src_plane_stride]+(D[1*src_plane_stride]-C[1*src_plane_stride])*dx;
        dst[1*out_plane_stride] = (h0+(h1-h0)*dy)*gain;

        h0 = A[2*src_plane_stride]+(B[2*src_plane_stride]-A[2*src_plane_stride])*dx;
        h1 = C[2*src_plane_stride]+(D[2*src_plane_stride]-C[2*src_plane_stride])*dx;
        dst[2*out_plane_stride] = (h0+(h1-h0)*dy)*gain;

        if (add_residue)
        {
            dst[0*out_plane_stride] += raw[0*raw_plane_stride];
            dst[1*out_plane_stride] += raw[1*raw_plane_stride];
            dst[2*out_plane_stride] += raw[2*raw_plane_stride];
        }
    }

    const auto time_warp_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_warp = (time_warp_end-time_warp_start);
    time_warp_seconds += (float)elapsed_warp.count();

    struct frame_memory result;
    result.raw_frame = out;
    result.origin_idx = 0;
    result.h = ref.h;
    result.w = ref.w;
    result.stride = ref.w;
    result.plane_stride = ref.h*ref.w;

    return result;
}

// final b-prediction
// add two pre-multiplied preds, multiply by alpha, add residue.
struct frame_memory bpred(struct frame_memory &raw_info, int raw_info_depth, struct frame_memory &pred0, struct frame_memory &pred1, int raw_gainidx)
{
    const auto time_bpred_start = std::chrono::steady_clock::now();
    int const h = raw_info.h;
    int const w = raw_info.w;
    float *raw = &raw_info.raw_frame[raw_info.origin_idx];
    int raw_stride = raw_info.stride;
    int raw_plane_stride = raw_info.plane_stride;

    float *out = new float[3*h*w]; // yuv444 = (pred1+pred2)*gain+residue, planar

    float *src0 = &pred0.raw_frame[pred0.origin_idx];
    float *src1 = &pred1.raw_frame[pred0.origin_idx];
    int const src_plane_stride = pred0.plane_stride;

    if (pred0.plane_stride != pred1.plane_stride || pred0.stride != pred0.w)
    {
        printf("internal error\n");
        printf("p0 stride %d width %d; p1 %d %d\n", pred0.stride, pred0.w, pred1.stride, pred1.w);
        exit(1);
    }
    float *dst = out; // first plane.
    int const raw_gain_offset = raw_gainidx * raw_plane_stride;

    for (int y = 0; y < h; y++, raw += raw_stride-w)
    for (int x = 0; x < w; x++, raw++, dst++, src0++, src1++)
    {
        float gain = raw[raw_gain_offset]+0.5;
        if (gain < 0.0)
            gain = 0.0;
        else if (gain > 1.0)
            gain = 1.0;

        dst[0*src_plane_stride] = (src0[0*src_plane_stride] +src1[0*src_plane_stride])*gain + raw[0*raw_plane_stride];
        dst[1*src_plane_stride] = (src0[1*src_plane_stride] +src1[1*src_plane_stride])*gain + raw[1*raw_plane_stride];
        dst[2*src_plane_stride] = (src0[2*src_plane_stride] +src1[2*src_plane_stride])*gain + raw[2*raw_plane_stride];
    }

    const auto time_bpred_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_bpred = (time_bpred_end-time_bpred_start);
    time_bpred_seconds += (float)elapsed_bpred.count();

    struct frame_memory result;
    result.raw_frame = out;
    result.origin_idx = 0;
    result.h = h;
    result.w = w;
    result.stride = w;
    result.plane_stride = h*w;

    return result;
}

// returns a yuv444 float, planar.
// raw_cc_output is either [3], [6] or [9] channel (intra, P, B)
// refx_444: float, planar.
struct frame_memory process_inter(struct frame_memory &raw_cc_output, int im_h, int im_w, struct frame_memory *ref0_444, struct frame_memory *ref1_444, int flo_gain)
{
    if (ref1_444 == NULL)
    {
        // P
        // multiplies by alpha, adds residue.
        struct frame_memory pred = warp(raw_cc_output, 6, *ref0_444, 3, 5, flo_gain, true);
        return pred;
    }
    else
    {
        // B
        // want to multiply by beta, do not add residue.
        struct frame_memory pred0 = warp(raw_cc_output, 9, *ref0_444, 3, 8, flo_gain, false);
        // want to multiply by 1-beta, do not add residue.
        struct frame_memory pred1 = warp(raw_cc_output, 9, *ref1_444, 6, -8, flo_gain, false);

        // want to add pred1, pred2, multiply by alpha, and add residue.
        struct frame_memory pred = bpred(raw_cc_output, 9, pred0, pred1, 5);
        delete[] pred0.raw_frame;
        delete[] pred1.raw_frame;

        return pred;
    }
}

#if 0
char const *param_bitstream = "--bitstream=";
char const *param_out       = "--out=";
char const *param_arm_type = "--arm=";
char const *param_ups_type = "--ups=";
char const *param_syn_type = "--syn=";

void usage(char const *msg = NULL)
{
    if (msg != NULL)
        printf("%s\n", msg);
    printf("Usage:\n");
    printf("    %s: .hevc to decode\n", param_bitstream);
    printf("    %s: reconstruction if desired: ppm (image) or yuv420 (video) only\n", param_out);
    printf("    %s: ARM processing type, 'cpu', 'avx2', or 'auto'\n", param_arm_type);
    printf("    %s: UPS processing type, 'cpu', 'avx2', or 'auto'\n", param_ups_type);
    printf("    %s: SYN processing type, 'cpu', 'avx2', or 'auto'\n", param_syn_type);
    exit(msg != NULL);
}

int main(int argc, const char* argv[])
{

    std::string bitstream_filename;
    std::string out = "out";
    std::string arm_type;
    std::string ups_type;
    std::string syn_type;

    for (int i = 1; i < argc; i++)
    {
        if (strncmp(argv[i], param_bitstream, strlen(param_bitstream)) == 0)
            bitstream_filename = argv[i]+strlen(param_bitstream);
        else if (strncmp(argv[i], param_out, strlen(param_out)) == 0)
            out = argv[i]+strlen(param_out);
        else if (strncmp(argv[i], param_arm_type, strlen(param_arm_type)) == 0)
            arm_type = argv[i]+strlen(param_arm_type);
        else if (strncmp(argv[i], param_ups_type, strlen(param_ups_type)) == 0)
            ups_type = argv[i]+strlen(param_ups_type);
        else if (strncmp(argv[i], param_syn_type, strlen(param_syn_type)) == 0)
            syn_type = argv[i]+strlen(param_syn_type);
        else
        {
            std::string error_message = "unknown parameter ";
            error_message += argv[i];
            usage(error_message.c_str());
        }
    }

    if (bitstream_filename == "")
        usage("must specify a bitstream");

    if (arm_type == "")
        arm_type = "auto";
    if (arm_type != "cpu" && arm_type != "avx2" && arm_type != "auto")
        usage("unknown arm_type");
    if (ups_type == "")
        ups_type = "auto";
    if (ups_type != "cpu" && ups_type != "avx2" && ups_type != "auto")
        usage("unknown ups_type");
    if (syn_type == "")
        syn_type = "auto";
    if (syn_type != "cpu" && syn_type != "avx2" && syn_type != "auto")
        usage("unknown syn_type");

#ifdef __APPLE__

    arm_type = "cpu";
    ups_type = "cpu";
    syn_type = "cpu";

#else

    std::string auto_type = "cpu";
    if (__builtin_cpu_supports("avx2"))
        auto_type = "avx2";

    if (arm_type == "auto")
        arm_type = auto_type;
    if (ups_type == "auto")
        ups_type = auto_type;
    if (syn_type == "auto")
        syn_type = auto_type;

#endif

    printf("running with ARM avx support: %s\n", arm_type.c_str());
    printf("running with UPS avx support: %s\n", ups_type.c_str());
    printf("running with SYN avx support: %s\n", syn_type.c_str());
    const auto time_all_start = std::chrono::steady_clock::now();

    cc_bs bs;
    if (!bs.open(bitstream_filename))
    {
        exit(1);
    }

    struct cc_bs_gop_header &gop_header = bs.m_gop_header;
    struct frame_memory frames_444[gop_header.intra_period+1]; // we store all decoded frames for the moment.

    memset(frames_444, 0, (gop_header.intra_period+1)*sizeof(frames_444[0]));

    FILE *fout = NULL;
    if (out != "")
    {
        fout = fopen(out.c_str(), "wb");
        if (fout == NULL)
        {
            printf("Cannot open %s for writing\n", out.c_str());
        }
    }

    for (int frame_coding_idx = 0; frame_coding_idx <= gop_header.intra_period; frame_coding_idx++)
    {
        // raw_cc_output either
        // [3] (I: residue)
        // [6] (P: residue+xy+alpha)
        // [9] (B: residue+xy+alpha+xy+beta)
        cc_bs_frame *frame_coded = bs.decode_frame();
        if (frame_coded == NULL)
        {
            exit(1);
        }
        struct frame_memory raw_cc_output = decode_frame(gop_header, frame_coded);
        struct cc_bs_frame_header &frame_header = frame_coded->m_frame_header;

        if (raw_cc_output.raw_frame == NULL)
        {
            printf("decoding failed\n");
            break;
        }

        struct frame_memory frame_444;
        unsigned char *frame_420;

        if (frame_coding_idx == 0)
        {
            // I
            frame_444 = raw_cc_output;
        }
        else
        {
            // search for references.
            // here, we distinguish P (syn: 6 outputs) and B (syn: 9 outputs)
            struct frame_memory *ref_prev = NULL;
            struct frame_memory *ref_next = NULL;
            // find prev for P and B.
            for (int idx = frame_header.display_index-1; idx >= 0; idx--)
                if (frames_444[idx].raw_frame != NULL)
                {
                    ref_prev = &frames_444[idx];
                    printf("refprev: %d\n", idx);
                    break;
                }
            // find next if B.
            if (frame_header.layers_synthesis[1].n_out_ft == 9)
                for (int idx = frame_header.display_index+1; idx <= gop_header.intra_period; idx++)
                    if (frames_444[idx].raw_frame != NULL)
                    {
                        ref_next = &frames_444[idx];
                        printf("refnext: %d\n", idx);
                        break;
                    }
            frame_444 = process_inter(raw_cc_output, gop_header.img_h, gop_header.img_w, ref_prev, ref_next, frame_header.flow_gain);
            delete [] raw_cc_output.raw_frame;
            raw_cc_output.raw_frame = NULL;
        }

        if (gop_header.frame_data_type == 1)
        {
            // 420
            frame_420 = convert_444_420_8b(frame_444);
            if (fout != NULL)
                dump_yuv420_8b(frame_420, gop_header.img_h, gop_header.img_w, fout, frame_header.display_index);
            frames_444[frame_header.display_index] = convert_420_444_8b(frame_420, gop_header.img_h, gop_header.img_w);
            delete[] frame_420;
        }
        else
        {
            // rgb
            if (out != "")
                ppm_out(3, 3, frame_444, out.c_str());
            frames_444[frame_header.display_index] = frame_444;
        }
        delete frame_coded;

        const auto time_all_end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed_all = (time_all_end-time_all_start);
        time_all_seconds = (float)elapsed_all.count();

        printf("time: arm %g ups %g syn %g warp %g bpred %g all %g\n",
                time_arm_seconds, time_ups_seconds, time_syn_seconds, time_warp_seconds, time_bpred_seconds, time_all_seconds);
        fflush(stdout);
    }

    if (fout != NULL)
        fclose(fout);

    printf("decode done\n");

    return 0;
}
#endif

#ifdef CCDECAPI_CPU
int cc_decode_cpu(std::string &bitstream_filename, std::string &out_filename)
#else
#ifdef CCDECAPI_AVX2
int cc_decode_avx2(std::string &bitstream_filename, std::string &out_filename)
#else
#error must have one of CCDECAPI_CPU or CCDECAPI_AVX2 defined.
#endif
#endif
{
    if (bitstream_filename == "")
    {
        printf("must specify a bitstream\n");
        return 1;
    }

    const auto time_all_start = std::chrono::steady_clock::now();

    cc_bs bs;
    if (!bs.open(bitstream_filename))
    {
        printf("cannot open %s for reading\n", bitstream_filename.c_str());
        return 1;
    }

    struct cc_bs_gop_header &gop_header = bs.m_gop_header;
    struct frame_memory frames_444[gop_header.intra_period+1]; // we store all decoded frames for the moment.

    memset(frames_444, 0, (gop_header.intra_period+1)*sizeof(frames_444[0]));

    FILE *fout = NULL;
    if (out_filename != "")
    {
        fout = fopen(out_filename.c_str(), "wb");
        if (fout == NULL)
        {
            printf("Cannot open %s for writing\n", out_filename.c_str());
            return 1;
        }
    }

    for (int frame_coding_idx = 0; frame_coding_idx <= gop_header.intra_period; frame_coding_idx++)
    {
        // raw_cc_output either
        // [3] (I: residue)
        // [6] (P: residue+xy+alpha)
        // [9] (B: residue+xy+alpha+xy+beta)
        cc_bs_frame *frame_coded = bs.decode_frame();
        if (frame_coded == NULL)
        {
            return 1;
        }
        struct frame_memory raw_cc_output = decode_frame(gop_header, frame_coded);
        struct cc_bs_frame_header &frame_header = frame_coded->m_frame_header;

        if (raw_cc_output.raw_frame == NULL)
        {
            printf("decoding failed\n");
            break;
        }

        struct frame_memory frame_444;

        if (frame_coding_idx == 0)
        {
            // I
            frame_444 = raw_cc_output;
        }
        else
        {
            // search for references.
            // here, we distinguish P (syn: 6 outputs) and B (syn: 9 outputs)
            struct frame_memory *ref_prev = NULL;
            struct frame_memory *ref_next = NULL;
            // find prev for P and B.
            for (int idx = frame_header.display_index-1; idx >= 0; idx--)
                if (frames_444[idx].raw_frame != NULL)
                {
                    ref_prev = &frames_444[idx];
                    printf("refprev: %d\n", idx);
                    break;
                }
            // find next if B.
            if (frame_header.layers_synthesis[1].n_out_ft == 9)
                for (int idx = frame_header.display_index+1; idx <= gop_header.intra_period; idx++)
                    if (frames_444[idx].raw_frame != NULL)
                    {
                        ref_next = &frames_444[idx];
                        printf("refnext: %d\n", idx);
                        break;
                    }
            frame_444 = process_inter(raw_cc_output, gop_header.img_h, gop_header.img_w, ref_prev, ref_next, frame_header.flow_gain);
            delete [] raw_cc_output.raw_frame;
            raw_cc_output.raw_frame = NULL;
        }

        if (gop_header.frame_data_type == 1)
        {
            // 420
            if (gop_header.bitdepth == 8)
            {
                unsigned char *frame_420;
                frame_420 = convert_444_420_8b(frame_444);
                if (fout != NULL)
                    dump_yuv420_8b(frame_420, gop_header.img_h, gop_header.img_w, fout, frame_header.display_index);
                frames_444[frame_header.display_index] = convert_420_444_8b(frame_420, gop_header.img_h, gop_header.img_w);
                delete[] frame_420;
            }
            if (gop_header.bitdepth == 10)
            {
                unsigned short *frame_420;
                frame_420 = convert_444_420_10b(frame_444);
                if (fout != NULL)
                    dump_yuv420_10b(frame_420, gop_header.img_h, gop_header.img_w, fout, frame_header.display_index);
                frames_444[frame_header.display_index] = convert_420_444_10b(frame_420, gop_header.img_h, gop_header.img_w);
                delete[] frame_420;
            }
            else
            {
                printf("Unkown bitdepth %d", gop_header.bitdepth);
            }
        }
        else
        {
            // rgb
            if (out_filename != "")
                ppm_out(3, 3, frame_444, out_filename.c_str());
            frames_444[frame_header.display_index] = frame_444;
        }
        delete frame_coded;

        const auto time_all_end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed_all = (time_all_end-time_all_start);
        time_all_seconds = (float)elapsed_all.count();

        printf("time: arm %g ups %g syn %g warp %g bpred %g all %g\n",
                time_arm_seconds, time_ups_seconds, time_syn_seconds, time_warp_seconds, time_bpred_seconds, time_all_seconds);
        fflush(stdout);
    }

    if (fout != NULL)
        fclose(fout);

    printf("decode done\n");

    return 0;
}
