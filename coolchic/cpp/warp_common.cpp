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
#include "warp_common.h"

// !!! global coeff storage for now.
int   g_coeff_table_motion_q_shift = -1;
int   g_coeff_table_ntaps = -1;
float *g_coeff_table = NULL; // index with [quantval * ntaps].

template <typename P>
inline P cubic_conv_near(P x, P A)
{               
    return ((A+2)*x - (A+3))*x*x + 1;
}           
            
template <typename P>
inline P cubic_conv_far(P x, P A)
{               
    return ((A*x - 5*A)*x + 8*A)*x - 4*A;
}       

template <typename P, int NTAPS>
inline void sinc_generate_coeffs(P coeffs[NTAPS], P t)
{           
    P pi_N = M_PI/NTAPS;

    P delta_i = NTAPS/2+t-1;
    for (int i = 1; i <= NTAPS; i++, delta_i -= 1.0)
    {
        if (delta_i != 0)
            coeffs[i-1] = cos(pi_N*delta_i)*sin(M_PI*delta_i)/(M_PI*delta_i);
        else
            coeffs[i-1] = 1.0;
    }
}           

template <typename P>
inline void bicubic_generate_coeffs(P coeffs[4], P t)
{           
    P tleft = t;
    P const A = -0.75;
    coeffs[0] = cubic_conv_far(tleft+1, A);
    coeffs[1] = cubic_conv_near(tleft, A);
    P tright = 1-t;
    coeffs[2] = cubic_conv_near(tright, A);
    coeffs[3] = cubic_conv_far(tright+1, A);
}           

template <typename P>
inline void bilinear_generate_coeffs(P coeffs[2], P t)
{           
    coeffs[0] = 1-t;
    coeffs[1] = t;
}           

template<typename P, int NTAPS>
void generate_coeffs(int motion_q_shift)
{
    if (g_coeff_table_motion_q_shift == motion_q_shift && g_coeff_table_ntaps == NTAPS)
        return; // nothing to do.

    // generate all coeffs for quantized motion.
    delete [] g_coeff_table;
    g_coeff_table = new float[NTAPS<<motion_q_shift];
    g_coeff_table_motion_q_shift = motion_q_shift;
    g_coeff_table_ntaps = NTAPS;

    for (int q = 0; q < (1<<motion_q_shift); q++)
    {
        switch (NTAPS)
        {
        case 2:
            bilinear_generate_coeffs<P>(&g_coeff_table[q*NTAPS], ((float)q)/(1<<motion_q_shift));
            break;
        case 4:
            bicubic_generate_coeffs<P>(&g_coeff_table[q*NTAPS], ((float)q)/(1<<motion_q_shift));
            break;
        default:
            sinc_generate_coeffs<P, NTAPS>(&g_coeff_table[q*NTAPS], ((float)q)/(1<<motion_q_shift));
            break;
        }
    }
}

// instantiate allowed ntaps.
template void generate_coeffs<float, 2>(int motion_q_shift);
template void generate_coeffs<float, 4>(int motion_q_shift);
template void generate_coeffs<float, 6>(int motion_q_shift);
template void generate_coeffs<float, 8>(int motion_q_shift);
template void generate_coeffs<float, 10>(int motion_q_shift);
template void generate_coeffs<float, 12>(int motion_q_shift);
template void generate_coeffs<float, 14>(int motion_q_shift);
template void generate_coeffs<float, 16>(int motion_q_shift);
