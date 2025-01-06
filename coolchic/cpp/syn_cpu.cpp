/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#include <stdio.h>
#include <memory.h>
#include <algorithm>

#include "common.h"
#include "syn_cpu.h" // basic defines and expected declarations.

#define SYN_NAME   custom_conv_ks1_inX_outX
#define SYN_KS     1
#include "syn_cpu.hpp"

#define SYN_NAME   custom_conv_ksX_inX_outX
#include "syn_cpu.hpp"

#define SYN_NAME   custom_conv_ks1_inX_hiddenX_outX
#define SYN_KS     1
#include "synfused_cpu.hpp"

#define SYN_NAME   custom_conv_ks3_inX_outX_lb
#define SYN_KS     3
#include "synlb_cpu.hpp"

#include "synblend_cpu.hpp"
