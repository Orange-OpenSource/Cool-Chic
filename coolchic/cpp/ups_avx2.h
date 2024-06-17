/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


void ups_4x4x4_avx2(int KS, float *kw, int h_in, int w_in, float *in, float *out);
void ups_4x2x2_avx2(int KS, float *kw, int h_in, int w_in, float *in, float *out);
