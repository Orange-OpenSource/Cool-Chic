/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


void custom_conv_ks1_in7_out9_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);
void custom_conv_ks1_in9_out3_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);
void custom_conv_ks1_in9_out6_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);
void custom_conv_ks1_in9_out9_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);
void custom_conv_ks1_inX_outX_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);

void custom_conv_ks1_in7_out40_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);
void custom_conv_ks1_in7_out16_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);
void custom_conv_ks1_inX_out3_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);
void custom_conv_ks1_inX_out6_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);
void custom_conv_ks1_inX_out9_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);

void custom_conv_ksX_inX_outX_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int residue, int relu);

void custom_conv_ks1_in7_hidden40_out3_avx2(int KS, int32_t *kw7_40, int32_t *kb40, int32_t *kw40_3, int32_t *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, int32_t *in, int N_OUT, int32_t *out);
void custom_conv_ks1_in7_hidden40_out6_avx2(int KS, int32_t *kw7_40, int32_t *kb40, int32_t *kw40_3, int32_t *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, int32_t *in, int N_OUT, int32_t *out);
void custom_conv_ks1_in7_hidden40_out9_avx2(int KS, int32_t *kw7_40, int32_t *kb40, int32_t *kw40_3, int32_t *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, int32_t *in, int N_OUT, int32_t *out);
void custom_conv_ks1_in7_hidden16_out3_avx2(int KS, int32_t *kw7_40, int32_t *kb40, int32_t *kw40_3, int32_t *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, int32_t *in, int N_OUT, int32_t *out);
void custom_conv_ks1_in7_hidden8_out3_avx2(int KS, int32_t *kw7_40, int32_t *kb40, int32_t *kw40_3, int32_t *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, int32_t *in, int N_OUT, int32_t *out);

void custom_conv_ks3_in3_out3_lb_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int32_t *line_buffer, int residue, int relu);
void custom_conv_ks3_in6_out6_lb_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int32_t *line_buffer, int residue, int relu);
void custom_conv_ks3_in9_out6_lb_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int32_t *line_buffer, int residue, int relu);
void custom_conv_ks3_in9_out9_lb_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int32_t *line_buffer, int residue, int relu);
void custom_conv_ks3_inX_outX_lb_avx2(int KS, int32_t *kw, int32_t *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, int32_t *in, int N_OUT, int32_t *out, int32_t *line_buffer, int residue, int relu);
