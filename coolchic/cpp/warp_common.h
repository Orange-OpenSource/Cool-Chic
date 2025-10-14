/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#include <stdlib.h>
#include <chrono> // timing.
#include <math.h>

#include "common.h"
#include "frame-memory.h"

extern float time_warp_seconds;
extern float time_warp_coeffgen_seconds;
extern float time_warp_coeffget_seconds;

inline int get_limited_l(int v_start_x, int x_idx)
{
    if (x_idx < v_start_x)
        x_idx = v_start_x;
    return x_idx;
}

inline int get_limited_r(int v_limit_x, int x_idx)
{
    if (x_idx >= v_limit_x-1)
        x_idx = v_limit_x-1;
    return x_idx;
}

inline int get_limited(int v_start_x, int v_limit_x, int x_idx)
{
    x_idx = get_limited_l(v_start_x, x_idx);
    x_idx = get_limited_r(v_limit_x, x_idx);
    return x_idx;
}
            
// !!! global coeff storage for now.
extern int   g_coeff_table_motion_q_shift;
extern int   g_coeff_table_ntaps;
extern float *g_coeff_table; // index with [quantval * ntaps].

template<typename P, int NTAPS>
void generate_coeffs(int motion_q_shift);

template<typename P, int NTAPS>
inline P *get_coeffs(int motion_qsteps)
{
    return &g_coeff_table[motion_qsteps*NTAPS];
}

// return basex and tx (integers) from a float px.
template <typename P>
inline void get_motion_info(P px, int motion_q_shift, int32_t &basex, int32_t &tx)
{
    basex = (int32_t)rintf(px*(1<<motion_q_shift));
    tx = basex&((1<<motion_q_shift)-1);

    if (basex < 0)
    {
        basex = -basex+((1<<motion_q_shift)-1);
        basex >>= motion_q_shift;
        basex = -basex;
    }
    else
    {
        basex >>= motion_q_shift;
    }
}
