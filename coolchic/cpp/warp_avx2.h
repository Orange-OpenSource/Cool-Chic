/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

template <typename P, typename PACC>
void warp_ntaps_avx2(struct frame_memory<P> &warp_result,
                     struct frame_memory<P> const &frame_residual, struct frame_memory<P> const &frame_motion, struct frame_memory<P> const &ref, int global_flow[2],
                     int raw_xyidx, int raw_gainidx, bool add_residue, int ntaps, int motion_blksz_shift, int motion_q);
