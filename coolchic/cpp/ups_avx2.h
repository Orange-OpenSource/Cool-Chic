/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


template <typename P>
void ups_refine_ks7_avx2(int ks, P *kw, frame_memory<P> &in, frame_memory<P> &out, int ups_src_precision, frame_memory<P> &tmp);
template <typename P>
void ups_refine_ksX_avx2(int ks, P *kw, frame_memory<P> &in, frame_memory<P> &out, int ups_src_precision, frame_memory<P> &tmp);
template <typename P>
void ups_upsample_ks8_UPSPREC_avx2(int ksx2, P *kw, frame_memory<P> &in, frame_memory<P> &out, int out_plane, int ups_src_precision, frame_memory<P> &tmp);
template <typename P>
void ups_upsample_ks8_ARMPREC_avx2(int ksx2, P *kw, frame_memory<P> &in, frame_memory<P> &out, int out_plane, int ups_src_precision, frame_memory<P> &tmp);
template <typename P>
void ups_upsample_ksX_avx2(int ksx2, P *kw, frame_memory<P> &in, frame_memory<P> &out, int out_plane, int ups_src_precision, frame_memory<P> &tmp);
