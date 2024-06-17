/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#include <stdio.h>
#include <memory.h>
#include <immintrin.h>
#include <algorithm>

// ks free, stride==1

#define SYN_NAME   custom_conv_ks1_in7_out9_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_OUT  9
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in9_out3_avx2
#define SYN_KS     1
#define SYN_N_IN   9
#define SYN_N_OUT  3
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in9_out6_avx2
#define SYN_KS     1
#define SYN_N_IN   9
#define SYN_N_OUT  6
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in9_out9_avx2
#define SYN_KS     1
#define SYN_N_IN   9
#define SYN_N_OUT  9
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_inX_outX_avx2
#define SYN_KS     1
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks3_in3_out3_avx2
#define SYN_KS     3
#define SYN_N_IN   3
#define SYN_N_OUT  3
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks3_in9_out6_avx2
#define SYN_KS     3
#define SYN_N_IN   9
#define SYN_N_OUT  6
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ksX_inX_outX_avx2
#include "syn_avx2.hpp"

