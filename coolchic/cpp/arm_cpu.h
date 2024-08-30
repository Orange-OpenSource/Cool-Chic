/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

void custom_conv_11_int32_cpu_X_X_X(weights_biases *kwtX_n_n, weights_biases *kbX_n, // kwt0_16_16 -- kernel weights, transposed.
                                    weights_biases *kwOUT_n_2, weights_biases *kbOUT_2,
                                    int32_t *context_indicies, int32_t n_contexts, int n_hidden_layers,
                                    int32_t *SRC,
                                    int src_h, int src_w, int src_pad,
                                    BACContext &bac_context
                                    );
