/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

//void custom_conv_11_int32_avx2_16_16_16(int32_t *kwt0_16_16, int32_t *kb0_16, // kwt0_16_16 -- kernel weights, transposed.
//                                        int32_t *kwt1_16_16, int32_t *kb1_16, // kwt1_16_16 -- kernel weights, transposed,
//                                        int32_t *kw2_16_2, int32_t *kb2_2,
//                                        int32_t *context_indicies,
//                                        int32_t *SRC,
//                                        int src_h, int src_w, int src_pad,
//                                        BACContext &bac_context
//                                        );
void custom_conv_11_int32_avx2_8_X_X(int32_t **kwtX_n_n, int32_t **kbX_n, // kwtX_n_n[n_hidden_layers] -- kernel weights, transposed.
                                     int32_t *kwOUT_n_2, int32_t *kbOUT_2, // _n_2, weights not transposed.
                                     int32_t *context_indicies, int32_t n_contexts_param, int32_t n_hidden_layers_param,
                                     int32_t *src,
                                     int src_h, int src_w, int src_pad,
                                     BACContext &bac_context
                                     );
void custom_conv_11_int32_avx2_16_X_X(int32_t **kwtX_n_n, int32_t **kbX_n, // kwtX_n_n[n_hidden_layers] -- kernel weights, transposed.
                                      int32_t *kwOUT_n_2, int32_t *kbOUT_2, // _n_2, weights not transposed.
                                      int32_t *context_indicies, int32_t n_contexts_param, int32_t n_hidden_layers_param,
                                      int32_t *src,
                                      int src_h, int src_w, int src_pad,
                                      BACContext &bac_context
                                      );
void custom_conv_11_int32_avx2_24_X_X(int32_t **kwtX_n_n, int32_t **kbX_n, // kwtX_n_n[n_hidden_layers] -- kernel weights, transposed.
                                      int32_t *kwOUT_n_2, int32_t *kbOUT_2, // _n_2, weights not transposed.
                                      int32_t *context_indicies, int32_t n_contexts_param, int32_t n_hidden_layers_param,
                                      int32_t *src,
                                      int src_h, int src_w, int src_pad,
                                      BACContext &bac_context
                                      );
void custom_conv_11_int32_avx2_32_X_X(int32_t **kwtX_n_n, int32_t **kbX_n, // kwtX_n_n[n_hidden_layers] -- kernel weights, transposed.
                                      int32_t *kwOUT_n_2, int32_t *kbOUT_2, // _n_2, weights not transposed.
                                      int32_t *context_indicies, int32_t n_contexts_param, int32_t n_hidden_layers_param,
                                      int32_t *src,
                                      int src_h, int src_w, int src_pad,
                                      BACContext &bac_context
                                      );
