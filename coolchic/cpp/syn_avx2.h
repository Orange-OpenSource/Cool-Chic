/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


template <typename P>
void custom_conv_ks1_in7_out9_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, int residue, int relu);
template <typename P>
void custom_conv_ks1_in9_out3_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, int residue, int relu);
template <typename P>
void custom_conv_ks1_in9_out6_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, int residue, int relu);
template <typename P>
void custom_conv_ks1_in9_out9_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, int residue, int relu);
template <typename P>
void custom_conv_ks1_inX_outX_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, int residue, int relu);

template <typename P>
void custom_conv_ks1_in7_out40_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, int residue, int relu);
template <typename P>
void custom_conv_ks1_in7_out16_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, int residue, int relu);
template <typename P>
void custom_conv_ks1_inX_out3_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, int residue, int relu);
template <typename P>
void custom_conv_ks1_inX_out6_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, int residue, int relu);
template <typename P>
void custom_conv_ks1_inX_out9_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, int residue, int relu);

template <typename P>
void custom_conv_ksX_inX_outX_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, int residue, int relu);

template <typename P>
void custom_conv_ks1_in7_hidden48_out3_avx2(int KS, P *kw7_40, P *kb40, P *kw40_3, P *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, P *in, int N_OUT, P *out);
template <typename P>
void custom_conv_ks1_in7_hidden40_out3_avx2(int KS, P *kw7_40, P *kb40, P *kw40_3, P *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, P *in, int N_OUT, P *out);
template <typename P>
void custom_conv_ks1_in7_hidden40_out6_avx2(int KS, P *kw7_40, P *kb40, P *kw40_3, P *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, P *in, int N_OUT, P *out);
template <typename P>
void custom_conv_ks1_in7_hidden40_out9_avx2(int KS, P *kw7_40, P *kb40, P *kw40_3, P *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, P *in, int N_OUT, P *out);
template <typename P>
void custom_conv_ks1_in7_hidden32_out3_avx2(int KS, P *kw7_40, P *kb40, P *kw40_3, P *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, P *in, int N_OUT, P *out);
template <typename P>
void custom_conv_ks1_in7_hidden16_out3_avx2(int KS, P *kw7_40, P *kb40, P *kw40_3, P *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, P *in, int N_OUT, P *out);
template <typename P>
void custom_conv_ks1_in7_hidden16_out4_avx2(int KS, P *kw7_40, P *kb40, P *kw40_3, P *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, P *in, int N_OUT, P *out);
template <typename P>
void custom_conv_ks1_in7_hidden16_out5_avx2(int KS, P *kw7_40, P *kb40, P *kw40_3, P *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, P *in, int N_OUT, P *out);
template <typename P>
void custom_conv_ks1_in7_hidden8_out3_avx2(int KS, P *kw7_40, P *kb40, P *kw40_3, P *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, P *in, int N_OUT, P *out);
template <typename P>
void custom_conv_ks1_in7_hidden8_out4_avx2(int KS, P *kw7_40, P *kb40, P *kw40_3, P *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, P *in, int N_OUT, P *out);
template <typename P>
void custom_conv_ks1_in7_hidden8_out5_avx2(int KS, P *kw7_40, P *kb40, P *kw40_3, P *kb3, int h_in, int w_in, int pad_in, int plane_stride_in, int N_IN, int N_HIDDEN, P *in, int N_OUT, P *out);

template <typename P>
void custom_conv_ks3_in3_out3_lb_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, P *line_buffer, int residue, int relu);
template <typename P>
void custom_conv_ks3_in4_out4_lb_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, P *line_buffer, int residue, int relu);
template <typename P>
void custom_conv_ks3_in5_out5_lb_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, P *line_buffer, int residue, int relu);
template <typename P>
void custom_conv_ks3_in6_out6_lb_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, P *line_buffer, int residue, int relu);
template <typename P>
void custom_conv_ks3_in9_out6_lb_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, P *line_buffer, int residue, int relu);
template <typename P>
void custom_conv_ks3_in9_out9_lb_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, P *line_buffer, int residue, int relu);
template <typename P>
void custom_conv_ks3_inX_outX_lb_avx2(int KS, P *kw, P *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, P *in, int N_OUT, P *out, P *line_buffer, int residue, int relu);

void syn_blend1_avx2(int h_in, int w_in, int stride_in, int plane_stride_in, int N_INOUT, int32_t *in, int32_t blend_val_in, int32_t *out);
void syn_blend2_avx2(int h_in, int w_in, int stride_in, int plane_stride_in, int N_INOUT, int32_t *in, int32_t blend_val_in, int32_t *out, int32_t blend_val_out);
