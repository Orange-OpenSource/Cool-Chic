/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#include <stdio.h>
#include <memory.h>
// #include <immintrin.h>   // Probably not useful
#include <algorithm>

// ks free, stride==1
#undef SYN_NAME
#undef SYN_KS
#undef SYN_N_IN
#undef SYN_N_OUT

#define SYN_NAME   custom_conv_ks1_inX_outX
#define SYN_KS     1
#include "syn_cpu.hpp"
#undef SYN_NAME
#undef SYN_KS
#undef SYN_N_IN
#undef SYN_N_OUT

#define SYN_NAME   custom_conv_ksX_inX_outX
#include "syn_cpu.hpp"
#undef SYN_NAME
#undef SYN_KS
#undef SYN_N_IN
#undef SYN_N_OUT

