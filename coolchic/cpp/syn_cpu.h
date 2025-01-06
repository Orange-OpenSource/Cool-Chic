/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


void custom_conv_ks1_inX_outX(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);
void custom_conv_ksX_inX_outX(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);
void custom_conv_ks1_inX_hiddenX_outX(int KS, int32_t *kw7_40, int32_t *kb40, int32_t *kw40_3, int32_t *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, int32_t *in, int N_OUT, int32_t *out);

void custom_conv_ks3_inX_outX_lb(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int32_t *line_buffer, int residue, int relu);

void syn_blend1(int h_in, int w_in, int stride_in, int plane_stride_in, int N_INOUT, int32_t *in, int32_t blend_val_in, int32_t *out);
void syn_blend2(int h_in, int w_in, int stride_in, int plane_stride_in, int N_INOUT, int32_t *in, int32_t blend_val_in, int32_t *out, int32_t blend_val_out);
