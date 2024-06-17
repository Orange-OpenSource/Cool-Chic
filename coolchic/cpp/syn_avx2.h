/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


void custom_conv_ks1_in7_out9_avx2(int KS, float *kw, float *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, float *in, int N_OUT, float *out, int residue, int relu);
void custom_conv_ks1_in9_out3_avx2(int KS, float *kw, float *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, float *in, int N_OUT, float *out, int residue, int relu);
void custom_conv_ks1_in9_out6_avx2(int KS, float *kw, float *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, float *in, int N_OUT, float *out, int residue, int relu);
void custom_conv_ks1_in9_out9_avx2(int KS, float *kw, float *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, float *in, int N_OUT, float *out, int residue, int relu);
void custom_conv_ks1_inX_outX_avx2(int KS, float *kw, float *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, float *in, int N_OUT, float *out, int residue, int relu);

void custom_conv_ks3_in3_out3_avx2(int KS, float *kw, float *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, float *in, int N_OUT, float *out, int residue, int relu);
void custom_conv_ks3_in9_out6_avx2(int KS, float *kw, float *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, float *in, int N_OUT, float *out, int residue, int relu);

void custom_conv_ksX_inX_outX_avx2(int KS, float *kw, float *kb, int h_in, int w_in, int stride_in, int plane_stride_in, int residue_origin_offset, int N_IN, float *in, int N_OUT, float *out, int residue, int relu);
