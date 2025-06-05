/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#include <stdio.h>
#include <memory.h>
#include <immintrin.h>
#include <algorithm>

#include "syn_avx2.h" // expected declarations.

#include "common.h"
#include "common_avx2.h"

// ks free, stride==1

#define SYN_NAME   custom_conv_ks1_in7_out9_avx2
#define SYN_ATATIME 3
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_OUT  9
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in9_out3_avx2
#define SYN_ATATIME 3
#define SYN_KS     1
#define SYN_N_IN   9
#define SYN_N_OUT  3
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in9_out6_avx2
#define SYN_ATATIME 3
#define SYN_KS     1
#define SYN_N_IN   9
#define SYN_N_OUT  6
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in9_out9_avx2
#define SYN_ATATIME 3
#define SYN_KS     1
#define SYN_N_IN   9
#define SYN_N_OUT  9
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_inX_outX_avx2
#define SYN_KS     1
#include "syn_avx2.hpp"

// !!! test
#define SYN_NAME   custom_conv_ks1_in7_out40_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_OUT  40
#define SYN_ATATIME 4
#include "syn_avx2.hpp"

// !!! test
#define SYN_NAME   custom_conv_ks1_in7_out16_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_OUT  16
#define SYN_ATATIME 4
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_inX_out3_avx2
#define SYN_KS     1
#define SYN_N_OUT  3
#define SYN_ATATIME 3
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_inX_out6_avx2
#define SYN_KS     1
#define SYN_N_OUT  6
#define SYN_ATATIME 3
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_inX_out9_avx2
#define SYN_KS     1
#define SYN_N_OUT  9
#define SYN_ATATIME 3
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ksX_inX_outX_avx2
#include "syn_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in7_hidden48_out3_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_HIDDEN 48
#define SYN_N_OUT  3
#include "synfused_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in7_hidden40_out3_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_HIDDEN 40
#define SYN_N_OUT  3
#include "synfused_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in7_hidden40_out6_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_HIDDEN 40
#define SYN_N_OUT  6
#include "synfused_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in7_hidden40_out9_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_HIDDEN 40
#define SYN_N_OUT  9
#include "synfused_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in7_hidden32_out3_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_HIDDEN 32
#define SYN_N_OUT  3
#include "synfused_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in7_hidden16_out3_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_HIDDEN 16
#define SYN_N_OUT  3
#include "synfused_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in7_hidden16_out4_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_HIDDEN 16
#define SYN_N_OUT  4
#include "synfused_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in7_hidden16_out5_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_HIDDEN 16
#define SYN_N_OUT  5
#include "synfused_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in7_hidden8_out3_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_HIDDEN 8
#define SYN_N_OUT  3
#include "synfused_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in7_hidden8_out4_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_HIDDEN 8
#define SYN_N_OUT  4
#include "synfused_avx2.hpp"

#define SYN_NAME   custom_conv_ks1_in7_hidden8_out5_avx2
#define SYN_KS     1
#define SYN_N_IN   7
#define SYN_N_HIDDEN 8
#define SYN_N_OUT  5
#include "synfused_avx2.hpp"

#define SYN_NAME   custom_conv_ks3_in3_out3_lb_avx2
#define SYN_KS     3
#define SYN_N_IN   3
#define SYN_N_OUT  3
#include "synlb_avx2.hpp"

#define SYN_NAME   custom_conv_ks3_in4_out4_lb_avx2
#define SYN_KS     3
#define SYN_N_IN   4
#define SYN_N_OUT  4
#include "synlb_avx2.hpp"

#define SYN_NAME   custom_conv_ks3_in5_out5_lb_avx2
#define SYN_KS     3
#define SYN_N_IN   5
#define SYN_N_OUT  5
#include "synlb_avx2.hpp"

#define SYN_NAME   custom_conv_ks3_in6_out6_lb_avx2
#define SYN_KS     3
#define SYN_N_IN   6
#define SYN_N_OUT  6
#include "synlb_avx2.hpp"

#define SYN_NAME   custom_conv_ks3_in9_out6_lb_avx2
#define SYN_KS     3
#define SYN_N_IN   9
#define SYN_N_OUT  6
#include "synlb_avx2.hpp"

#define SYN_NAME   custom_conv_ks3_in9_out9_lb_avx2
#define SYN_KS     3
#define SYN_N_IN   9
#define SYN_N_OUT  9
#include "synlb_avx2.hpp"

#define SYN_NAME   custom_conv_ks3_inX_outX_lb_avx2
#define SYN_KS     3
#include "synlb_avx2.hpp"

#include "synblend_avx2.hpp"
