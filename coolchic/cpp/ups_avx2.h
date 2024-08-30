/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


void ups_4x4x4_fromarm_avx2(int KS, int32_t *kw, int h_in, int w_in, int stride_in, int32_t *in, int stride_out, int32_t *out);
void ups_4x4x4_fromups_avx2(int KS, int32_t *kw, int h_in, int w_in, int stride_in, int32_t *in, int stride_out, int32_t *out);
void ups_4x2x2_fromarm_avx2(int KS, int32_t *kw, int h_in, int w_in, int stride_int, int32_t *in, int stride_out, int32_t *out);
void ups_4x2x2_fromups_avx2(int KS, int32_t *kw, int h_in, int w_in, int stride_int, int32_t *in, int stride_out, int32_t *out);
