/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#include <stdio.h>
#include <vector>
#include <string>
#include "cc-bitstream.h"

int read_int_1(FILE *bs)
{
    unsigned char buf[1];
    if (fread(&buf[0], 1, 1, bs) != 1)
        return -1;
    return buf[0];
}

int read_int_2(FILE *bs)
{
    unsigned char buf[2];
    if (fread(&buf[0], 1, 2, bs) != 2)
        return -1;
    return (buf[0]<<8)|buf[1];
}

int read_int_3(FILE *bs)
{
    unsigned char buf[3];
    if (fread(&buf[0], 1, 3, bs) != 3)
        return -1;
    return (buf[0]<<16)|(buf[1]<<8)|buf[2];
}

int read_bitdepth(FILE *bs)
{
    int raw = read_int_1(bs);
    return (raw < 16) ? 8 : 10;
}

bool cc_bs::open(std::string filename, int verbosity)
{
    m_filename = filename;
    m_f = fopen(filename.c_str(), "rb");
    if (m_f == NULL)
    {
        printf("Cannot open bitstream file %s for reading.\n", filename.c_str());
        return false;
    }
    return read_gop_header(verbosity);
}

bool cc_bs::read_gop_header(int verbosity)
{
    int raw;

    m_gop_header.n_bytes_header = read_int_2(m_f);
    m_gop_header.img_h = read_int_2(m_f);
    m_gop_header.img_w = read_int_2(m_f);
    raw = read_int_1(m_f);
    //m_gop_header.bitdepth = (raw>>4) == 0 ? 8 : 10;
    m_gop_header.bitdepth = (raw>>4) + 8; // !!! ASSUMING 8, 9..16 as 1st indicies.
    m_gop_header.frame_data_type = raw&0xF;
    m_gop_header.intra_period = read_int_1(m_f);
    m_gop_header.p_period = read_int_1(m_f);

    if (verbosity >= 2)
    {
        printf("GOP HEADER:\n");
        printf("    n_bytes_header: %d\n", m_gop_header.n_bytes_header);
        printf("    img_size: %d,%d\n", m_gop_header.img_h, m_gop_header.img_w);
        printf("    frame_data_type: %d\n", m_gop_header.frame_data_type);
        printf("    bitdepth: %d\n", m_gop_header.bitdepth);
        printf("    intra_period: %d\n", m_gop_header.intra_period);
        printf("    p_period: %d\n", m_gop_header.p_period);
    }

    return true;
}

void read_syn_layer(FILE *bs, struct cc_bs_syn_layer &layer)
{
    int raw;

    layer.n_out_ft = read_int_1(bs);
    layer.ks = read_int_1(bs);

    // !!! limited here -- should be limited elsewhere.  needs checking.
    raw = read_int_1(bs);
    layer.residual = (bool)(raw>>4);
    layer.relu = (bool)(raw&0xF);
}

void read_q_step_index_nn(FILE *bs, struct cc_bs_layer_quant_info &lqi)
{
    lqi.q_step_index_nn_weight = read_int_1(bs);
    lqi.q_step_index_nn_bias = read_int_1(bs);
    if (lqi.q_step_index_nn_bias == 255)
        lqi.q_step_index_nn_bias = -1;
}

void read_scale_index_nn(FILE *bs, struct cc_bs_layer_quant_info &lqi)
{
    lqi.scale_index_nn_weight = read_int_1(bs);
    if (lqi.q_step_index_nn_bias < 0)
        lqi.scale_index_nn_bias = -1;
    else
        lqi.scale_index_nn_bias = read_int_1(bs);
}

void read_n_bytes_nn(FILE *bs, struct cc_bs_layer_quant_info &lqi)
{
    lqi.n_bytes_nn_weight = read_int_2(bs);
    if (lqi.q_step_index_nn_bias < 0)
        lqi.n_bytes_nn_bias = -1;
    else
        lqi.n_bytes_nn_bias = read_int_2(bs);
}

void print_lqi(char const *name, struct cc_bs_layer_quant_info &lqi)
{
    printf("    %s:", name);
    printf(" q_step_index w/b %d %d", lqi.q_step_index_nn_weight, lqi.q_step_index_nn_bias);
    printf(" q_scale_index w/b %d %d", lqi.scale_index_nn_weight, lqi.scale_index_nn_bias);
    printf(" n_bytes w/b %d %d", lqi.n_bytes_nn_weight, lqi.n_bytes_nn_bias);
    printf("\n");
}

void print_syn(struct cc_bs_syn_layer &syn)
{
    printf("out%d-ks%d-%s-%s",
           syn.n_out_ft, syn.ks, syn.residual ? "residual" : "linear", syn.relu ? "relu" : "none");
}

bool read_frame_header(FILE *bs, struct cc_bs_frame_header &frame_header, int verbosity)
{
    int raw;

    frame_header.n_bytes_header = read_int_2(bs);
    frame_header.display_index = read_int_1(bs);
    raw = read_int_1(bs);
    frame_header.dim_arm = 8*(raw>>4);
    frame_header.n_hidden_layers_arm = raw&0xF;

    raw = read_int_1(bs);
    frame_header.n_ups_kernel = raw>>4;
    frame_header.ups_k_size = raw&0xF;
    raw = read_int_1(bs);
    frame_header.n_ups_preconcat_kernel = raw>>4;
    frame_header.ups_preconcat_k_size = raw&0xF;

    frame_header.n_syn_branches = read_int_1(bs);
    frame_header.n_syn_layers = read_int_1(bs);
    frame_header.layers_synthesis.resize(frame_header.n_syn_layers);
    for (int i = 0; i < frame_header.n_syn_layers; i++)
    {
        read_syn_layer(bs, frame_header.layers_synthesis[i]);
    }

    frame_header.flow_gain = read_int_1(bs);
    frame_header.ac_max_val_nn = read_int_2(bs);
    frame_header.ac_max_val_latent = read_int_2(bs);
    frame_header.hls_sig_blksize = (signed char)read_int_1(bs); // signed.

    read_q_step_index_nn(bs, frame_header.arm_lqi);
    read_q_step_index_nn(bs, frame_header.ups_lqi);
    read_q_step_index_nn(bs, frame_header.syn_lqi);

    read_scale_index_nn(bs, frame_header.arm_lqi);
    read_scale_index_nn(bs, frame_header.ups_lqi);
    read_scale_index_nn(bs, frame_header.syn_lqi);

    read_n_bytes_nn(bs, frame_header.arm_lqi);
    read_n_bytes_nn(bs, frame_header.ups_lqi);
    read_n_bytes_nn(bs, frame_header.syn_lqi);

    frame_header.n_latent_n_resolutions = read_int_1(bs);
    frame_header.latent_n_2d_grid = read_int_1(bs);

    frame_header.n_ft_per_latent.resize(frame_header.n_latent_n_resolutions);
    for (int i = 0; i < frame_header.n_latent_n_resolutions; i++)
        frame_header.n_ft_per_latent[i] = read_int_1(bs);

    frame_header.n_bytes_per_latent.resize(frame_header.latent_n_2d_grid);
    for (int i = 0; i < frame_header.latent_n_2d_grid; i++)
        frame_header.n_bytes_per_latent[i] = read_int_3(bs);

    if (verbosity >= 2)
    {
        printf("FRAME HEADER\n");
        printf("    n_bytes_header: %d\n", frame_header.n_bytes_header);
        printf("    display_index: %d\n", frame_header.display_index);
        printf("    dim_arm: %d\n", frame_header.dim_arm);
        printf("    n_hidden_layers_arm: %d\n", frame_header.n_hidden_layers_arm);
        printf("    n_ups_kernel=%d, ups_ks=%d\n", frame_header.n_ups_kernel, frame_header.ups_k_size);
        printf("    n_ups_preconcat_kernel=%d, ups_preconcat_ks=%d\n", frame_header.n_ups_preconcat_kernel, frame_header.ups_preconcat_k_size);
        printf("    n_syn_branches: %d\n", frame_header.n_syn_branches);
        printf("    layers_synthesis:");
        for (int i = 0; i < frame_header.n_syn_layers; i++)
        {
            printf(" ");
            print_syn(frame_header.layers_synthesis[i]);
        }
        printf("\n");

        printf("    flow_gain: %d\n", frame_header.flow_gain);
        printf("    ac_max_val_nn: %d\n", frame_header.ac_max_val_nn);
        printf("    ac_max_val_latent: %d\n", frame_header.ac_max_val_latent);
        printf("    hls_sig_blksize: %d\n", frame_header.hls_sig_blksize);

        print_lqi("arm", frame_header.arm_lqi);
        print_lqi("ups", frame_header.ups_lqi);
        print_lqi("syn", frame_header.syn_lqi);

        printf("    latent_n_resolutions: %d\n", frame_header.n_latent_n_resolutions);
        printf("    latent_n_2d_grid: %d\n", frame_header.latent_n_2d_grid);

        printf("    n_ft_per_latent:");
        for (int i = 0; i < frame_header.n_latent_n_resolutions; i++)
            printf(" %d", frame_header.n_ft_per_latent[i]);
        printf("\n");
        printf("    n_bytes_per_latent:");
        for (int i = 0; i < frame_header.latent_n_2d_grid; i++)
            printf(" %d", frame_header.n_bytes_per_latent[i]);
        printf("\n");
    }
     
    return true;
}

// return bytes in a vector suitable for giving to cabac for decode.
std::vector<unsigned char> get_coded(FILE *bs, int n_bytes)
{
    std::vector<unsigned char> result(n_bytes);
    if ((int)fread(reinterpret_cast<char*>(result.data()), 1, n_bytes, bs) != n_bytes)
    {
        printf("failed to read hevc-coded data for %d bytes\n", n_bytes);
        exit(1);
    }

    return result;
}

struct cc_bs_frame *cc_bs::decode_frame(int verbosity)
{
    cc_bs_frame *result = new cc_bs_frame();
    if (!read_frame_header(m_f, result->m_frame_header, verbosity))
    {
        delete result;
        return NULL;
    }

    struct cc_bs_frame_header &frame_header = result->m_frame_header;
    result->m_arm_weights_hevc = get_coded(m_f, frame_header.arm_lqi.n_bytes_nn_weight);
    result->m_arm_biases_hevc = get_coded(m_f, frame_header.arm_lqi.n_bytes_nn_bias);
    result->m_ups_weights_hevc = get_coded(m_f, frame_header.ups_lqi.n_bytes_nn_weight);
    result->m_ups_biases_hevc = get_coded(m_f, frame_header.ups_lqi.n_bytes_nn_bias);
    result->m_syn_weights_hevc = get_coded(m_f, frame_header.syn_lqi.n_bytes_nn_weight);
    result->m_syn_biases_hevc = get_coded(m_f, frame_header.syn_lqi.n_bytes_nn_bias);

    //printf("arm coded w%ld b%ld bytes w:%02x %02x %02x\n", result->m_arm_weights_hevc.size(), result->m_arm_biases_hevc.size(), result->m_arm_weights_hevc[0], result->m_arm_weights_hevc[1], result->m_arm_weights_hevc[2]);
    //printf("ups coded w%ld b%ld bytes w:%02x %02x %02x\n", result->m_ups_weights_hevc.size(), result->m_ups_biases_hevc.size(), result->m_ups_weights_hevc[0], result->m_ups_weights_hevc[1], result->m_ups_weights_hevc[2]);
    //printf("syn coded w%ld b%ld bytes w %02x %02x %02x\n", result->m_syn_weights_hevc.size(), result->m_syn_biases_hevc.size(), result->m_syn_weights_hevc[0], result->m_syn_weights_hevc[1], result->m_syn_weights_hevc[2]);
    //fflush(stdout);

    for (int i = 0; i < frame_header.n_latent_n_resolutions; i++)
        result->m_latents_hevc.emplace_back(get_coded(m_f, frame_header.n_bytes_per_latent[i]));

    return result;
}
