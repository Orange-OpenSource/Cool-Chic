/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#if !defined(_COMMON_)

#define _COMMON_
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <vector>
#include <chrono>

int const ARM_MAX_N_FEATURES = 32;

// 64-alignment for possible avx512 use
#define ALIGNTO 64
#define ALIGN(val) ((val+ALIGNTO-1)/ALIGNTO*ALIGNTO)

#define ARM_PRECISION 8
#define ARM_SCALE (1<<ARM_PRECISION)

#define UPS_PRECISION 12

#define SYN_WEIGHT_PRECISION 12 // biases are 2x this.
#define SYN_LAYER_PRECISION 12
// in fact, we are assuming weight- and layer- precisions are equal.
#define SYN_MUL_PRECISION SYN_LAYER_PRECISION

// minimal cache around allocations for weights, biases and buffers.
struct buffer
{
    buffer(): data(NULL), n(0) {}
    ~buffer() { unuse(); }
    int32_t *data;
    int      n;
    int32_t *update_to(int new_n) { if (new_n <= n) return data;
                                    unuse();
                                    n = new_n;
                                    data = (int32_t *)aligned_alloc(ALIGNTO, ALIGN(n*sizeof(data[0])));
                                    if (data == NULL)
                                    {
                                        printf("Cannot allocate weight/bias data: %d elements", new_n);
                                        exit(1);
                                    }
                                    return data;
                                  }
private:
    void unuse() { if (data != NULL) {free(data); data = NULL;} }
};
using weights_biases = buffer;

#endif
