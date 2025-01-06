/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#include "TDecBinCoderCABAC.h"
#include "common.h"
#include "cc-contexts.h"
#include "cc-bac.h"
#include "arm_cpu.h"

// generic: nctxts (8,16,24,32), hiddenlayers (0,1,2)
// chains two 16_16, 16_16, final 16_2
void custom_conv_11_int32_cpu_X_X_X(weights_biases *kwtX_n_n, weights_biases *kbX_n, // kwtX_n_n[n_hidden_layers] -- kernel weights, transposed.
                                    weights_biases *kwOUT_n_2, weights_biases *kbOUT_2, // _n_2, weights not transposed.
                                    int32_t *context_indicies, int32_t n_contexts, int32_t n_hidden_layers,
                                    int32_t *SRC,
                                    int src_h, int src_w, int src_pad,
                                    BACContext &bac_context
                                    )
{
    int const n_inout = n_contexts;
    int const n_final_out = 2;

    int32_t *src = SRC;
    int32_t ioX[2][32]; // buffers used for input and output.

    for (int y = 0; y < src_h; y++, src += src_pad+src_pad) // pads are: eol of this, and bol of next.
    for (int x = 0; x < src_w; x++, src++)
    {
        if (!bac_coded(bac_context, y, x))
        {
            src[0] = 0;
            continue;
        }
        int use_left = 1;
        if (bac_flat(bac_context, y, x, use_left))
        {
            if (use_left)
                src[0] = src[-1];
            else
                src[0] = src[-(src_w+src_pad+src_pad)];
            continue;
        }

        int32_t *inputs = &ioX[0][0]; // switches in outputsX
        int32_t *outputs = &ioX[0][0]; // switches in outputsX

        // load input.
        for (int i = 0; i < n_inout; i++)
        {
            inputs[i] = src[context_indicies[i]]; // gather
        }

        for (int hl = 0; hl < n_hidden_layers; hl++)
        {
            inputs = &ioX[(hl+0)%2][0];
            outputs = &ioX[(hl+1)%2][0];
            // operate the first kwt.
            int32_t *kw = kwtX_n_n[hl].data;
            int32_t *kb = kbX_n[hl].data;

            for (int i = 0; i < n_inout; i++)
                outputs[i] = kb[i] + inputs[i]*ARM_SCALE; // residual == 1
            for (int il = 0; il < n_inout; il++, kw += n_inout)
            {
                for (int i = 0; i < n_inout; i++)
                    outputs[i] += inputs[il]*kw[i];
            }
            for (int i = 0; i < n_inout; i++)
            {
                if (outputs[i] < 0)
                    outputs[i] = 0;
                else
                    outputs[i] = (outputs[i]+(ARM_SCALE/2)) >> ARM_PRECISION;
            }
        }

        // FINAL 24 -> 2
        int32_t out[2];
        int32_t *kw = kwOUT_n_2->data;
        int32_t *kb = kbOUT_2->data;
        for (int ol = 0; ol < n_final_out; ol++, kw += n_inout)
        {
            int32_t sum = kb[ol];
            for (int il = 0; il < n_inout; il++)
                sum += outputs[il]*kw[il];
            if (sum < 0)
                sum = -((-sum+ARM_SCALE/2) >> ARM_PRECISION);
            else
                sum = (sum+ARM_SCALE/2) >> ARM_PRECISION;
            out[ol] = sum;
        }

        // bac it.
        int xx = decode_latent_layer_bac_single(
                        bac_context,
                        out[0], out[1]
                    );
        src[0] = xx<<ARM_PRECISION;
    } // x, y
}
