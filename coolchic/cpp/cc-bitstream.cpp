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

int32_t read_utf_coded(FILE *bs, bool has_sign = false)
{
    int32_t result = 0;
    unsigned char buf[1];
    int cnt = 0;

    do
    {
        if (cnt >= 5)
        {
            printf("utf_coding error: not-terminated after 5 bytes\n");
            exit(1);
        }
        if (fread(&buf[0], 1, 1, bs) != 1)
            return -1;
        result <<= 7;
        result |= (buf[0]&0x7f);
        cnt += 1;
    } while ((buf[0]&0x80) == 0x80);

    if (has_sign)
    {
        int signbit = result&0x01;
        result >>= 1;
        if (signbit != 0)
            result = -result;
    }

    return result;
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
    m_gop_header.n_bytes_header = read_int_1(m_f);
    m_gop_header.img_h = read_int_2(m_f);
    m_gop_header.img_w = read_int_2(m_f);
    int x = read_int_1(m_f);
    m_gop_header.frame_data_type = x >> 4; // rgb, yuv
    m_gop_header.output_bitdepth = (x&0xf) + 8;

    if (verbosity >= 2)
    {
        printf("GOP HEADER:\n");
        printf("    n_bytes_header: %d\n", m_gop_header.n_bytes_header);
        printf("    img_size: %d,%d\n", m_gop_header.img_h, m_gop_header.img_w);
        printf("    frame_data_type: %d\n", m_gop_header.frame_data_type);
        printf("    output_bitdepth: %d\n", m_gop_header.output_bitdepth);
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
    lqi.n_bytes_nn_weight = read_utf_coded(bs);
    if (lqi.q_step_index_nn_bias < 0)
        lqi.n_bytes_nn_bias = -1;
    else
        lqi.n_bytes_nn_bias = read_utf_coded(bs);
}

void print_lqi(char const *name, struct cc_bs_layer_quant_info &lqi)
{
    printf("    %s:", name);
    printf(" q_step_index w/b %d %d", lqi.q_step_index_nn_weight, lqi.q_step_index_nn_bias);
    printf(" q_scale_index w/b %d %d", lqi.scale_index_nn_weight, lqi.scale_index_nn_bias);
    printf(" n_bytes w/b %d %d", lqi.n_bytes_nn_weight, lqi.n_bytes_nn_bias);
    printf("\n");
}

static void print_syn(struct cc_bs_syn_layer &syn)
{
    printf("out%d-ks%d-%s-%s",
           syn.n_out_ft, syn.ks, syn.residual ? "residual" : "linear", syn.relu ? "relu" : "none");
}

void cc_bs_frame_coolchic::copy_archi_from(cc_bs_frame_coolchic const &src, int topology_copy_bits)
{
    if (topology_copy_bits&(1<<TOP_COPY_ARM))
    {
        n_hidden_layers_arm = src.n_hidden_layers_arm;
        dim_arm = src.dim_arm;
    }

    if (topology_copy_bits&(1<<TOP_COPY_UPS))
    {
        n_ups_kernel = src.n_ups_kernel;
        ups_k_size = src.ups_k_size;
        n_ups_preconcat_kernel = src.n_ups_preconcat_kernel;
        ups_preconcat_k_size = src.ups_preconcat_k_size;
    }

    if (topology_copy_bits&(1<<TOP_COPY_SYN))
    {
        n_syn_layers = src.n_syn_layers;
        layers_synthesis = src.layers_synthesis;
    }

    if (topology_copy_bits&(1<<TOP_COPY_LAT))
    {
        n_latent_n_resolutions = src.n_latent_n_resolutions;
        latent_n_2d_grid = src.latent_n_2d_grid;
        n_ft_per_latent = src.n_ft_per_latent;
    }
}

static bool read_cc_archi(struct cc_bs_frame_coolchic &out, int topology_copy_bits, FILE *bs)
{
    int raw;

    if ((topology_copy_bits&(1<<TOP_COPY_ARM)) == 0)
    {
        raw = read_int_1(bs);
        out.dim_arm = 8*(raw>>4);
        out.n_hidden_layers_arm = raw&0xF;
    }

    if ((topology_copy_bits&(1<<TOP_COPY_UPS)) == 0)
    {
        raw = read_int_1(bs);
        out.n_ups_kernel = raw>>4;
        out.ups_k_size = raw&0xF;
        raw = read_int_1(bs);
        out.n_ups_preconcat_kernel = raw>>4;
        out.ups_preconcat_k_size = raw&0xF;
    }

    if ((topology_copy_bits&(1<<TOP_COPY_SYN)) == 0)
    {
        out.n_syn_layers = read_int_1(bs);
        out.layers_synthesis.resize(out.n_syn_layers);
        for (int i = 0; i < out.n_syn_layers; i++)
        {
            read_syn_layer(bs, out.layers_synthesis[i]);
        }
    }

    if ((topology_copy_bits&(1<<TOP_COPY_LAT)) == 0)
    {
        out.n_latent_n_resolutions = read_int_1(bs);
        out.latent_n_2d_grid = 0;

        out.n_ft_per_latent.resize(out.n_latent_n_resolutions);
        for (int i = 0; i < out.n_latent_n_resolutions; i++)
        {
            out.n_ft_per_latent[i] = read_int_1(bs);
            if (out.n_ft_per_latent[i] != 0 && out.n_ft_per_latent[i] != 1)
            {
                printf("unsupported: #ft for latent resolution %d: %d; only 0 or 1 are supported\n", i, out.n_ft_per_latent[i]);
                printf("printing partial coolchic\n");
                out.print();
                exit(1);
            }
            out.latent_n_2d_grid += out.n_ft_per_latent[i];
        }
    }

    return true;
}

// quantization and CABAC substream lengths.
static bool read_cc_lengths(struct cc_bs_frame_coolchic &cc, bool latents_zero, FILE *bs)
{
    cc.latents_zero = latents_zero;
    cc.hls_sig_blksize = (signed char)read_int_1(bs); // signed.

    read_q_step_index_nn(bs, cc.arm_lqi);
    if (!cc.latents_zero)
        read_q_step_index_nn(bs, cc.ups_lqi);
    read_q_step_index_nn(bs, cc.syn_lqi);

    read_scale_index_nn(bs, cc.arm_lqi);
    if (!cc.latents_zero)
        read_scale_index_nn(bs, cc.ups_lqi);
    read_scale_index_nn(bs, cc.syn_lqi);

    read_n_bytes_nn(bs, cc.arm_lqi);
    if (!cc.latents_zero)
        read_n_bytes_nn(bs, cc.ups_lqi);
    read_n_bytes_nn(bs, cc.syn_lqi);

    cc.n_bytes_per_latent.resize(cc.n_latent_n_resolutions);
    for (int i = 0; i < cc.n_latent_n_resolutions; i++)
    {
        if (cc.latents_zero)
            cc.n_bytes_per_latent[i] = 0;
        else if (cc.n_ft_per_latent[i] == 0)
            cc.n_bytes_per_latent[i] = 0;
        else
            cc.n_bytes_per_latent[i] = read_utf_coded(bs);
    }

    return true;
}

void cc_bs_frame_coolchic::print()
{
    printf("coolchic-archi:\n");
    printf("    dim_arm: %d\n", dim_arm);
    printf("    n_hidden_layers_arm: %d\n", n_hidden_layers_arm);
    printf("    n_ups_kernel=%d, ups_ks=%d\n", n_ups_kernel, ups_k_size);
    printf("    n_ups_preconcat_kernel=%d, ups_preconcat_ks=%d\n", n_ups_preconcat_kernel, ups_preconcat_k_size);
    printf("    layers_synthesis:");
    for (int i = 0; i < n_syn_layers; i++)
    {
        printf(" ");
        print_syn(layers_synthesis[i]);
    }
    printf("\n");

    printf("    hls_sig_blksize: %d\n", hls_sig_blksize);

    print_lqi("arm", arm_lqi);
    if (!latents_zero)
        print_lqi("ups", ups_lqi);
    print_lqi("syn", syn_lqi);

    printf("    latent_n_resolutions: %d\n", n_latent_n_resolutions);
    printf("    latent_n_2d_grid: %d\n", latent_n_2d_grid);

    printf("    n_ft_per_latent:");
    for (auto &v: n_ft_per_latent)
        printf(" %d", v);
    printf("\n");
    printf("    n_bytes_per_latent:");
    for (auto &v: n_bytes_per_latent)
    {
        printf(" %d", v);
        fflush(stdout);
    }
    printf("\n");
}

static bool read_frame_header(FILE *bs, struct cc_bs_frame_header &frame_header, int verbosity)
{
    frame_header.n_bytes_header = read_utf_coded(bs);
    frame_header.display_index = read_utf_coded(bs);

    int bits = read_int_1(bs);
    switch (bits&0x3)
    {
    case 0: frame_header.frame_type = 'I';
            bits >>= 2;
            frame_header.latents_zero[0] = (bits&0x1) != 0;
            frame_header.latents_zero[1] = 0;
            break;

    case 1: frame_header.frame_type = 'P';
            bits >>= 2;
            frame_header.latents_zero[0] = (bits&0x1) != 0;
            bits >>= 1;
            frame_header.topology_copy[0] = (bits&0xf);

            bits = read_int_1(bs);
            frame_header.latents_zero[1] = (bits&0x1) != 0;
            bits >>= 1;
            frame_header.topology_copy[1] = (bits&0xf);
            break;

    case 2: frame_header.frame_type = 'B';
            bits >>= 2;
            frame_header.latents_zero[0] = (bits&0x1) != 0;
            bits >>= 1;
            frame_header.topology_copy[0] = (bits&0xf);

            bits = read_int_1(bs);
            frame_header.latents_zero[1] = (bits&0x1) != 0;
            bits >>= 1;
            frame_header.topology_copy[1] = (bits&0xf);
            break;

    default: return false; // perhaps EOF.
    }

    if (frame_header.frame_type == 'P' || frame_header.frame_type == 'B')
    {
        frame_header.global_flow[0][0] = read_utf_coded(bs, true);
        frame_header.global_flow[0][1] = read_utf_coded(bs, true);
        if (frame_header.frame_type == 'B')
        {
            frame_header.global_flow[1][0] = read_utf_coded(bs, true);
            frame_header.global_flow[1][1] = read_utf_coded(bs, true);
        }
    }

    if (feof(bs))
        return false;

    return true;
}

// return bytes in a vector suitable for giving to cabac for decode.
static std::vector<unsigned char> get_coded(FILE *bs, int n_bytes)
{
    std::vector<unsigned char> result(n_bytes);
    if ((int)fread(reinterpret_cast<char*>(result.data()), 1, n_bytes, bs) != n_bytes)
    {
        printf("failed to read hevc-coded data for %d bytes\n", n_bytes);
        exit(1);
    }

    return result;
}

// update the coolchic data with the cabac compressed weights, biases and latents.
static bool read_cc_content(struct cc_bs_frame_coolchic &cc, FILE *f)
{

    cc.m_arm_weights_hevc = get_coded(f, cc.arm_lqi.n_bytes_nn_weight);
    cc.m_arm_biases_hevc = get_coded(f, cc.arm_lqi.n_bytes_nn_bias);
    cc.m_ups_weights_hevc = get_coded(f, cc.ups_lqi.n_bytes_nn_weight);
    cc.m_ups_biases_hevc = get_coded(f, cc.ups_lqi.n_bytes_nn_bias);
    cc.m_syn_weights_hevc = get_coded(f, cc.syn_lqi.n_bytes_nn_weight);
    cc.m_syn_biases_hevc = get_coded(f, cc.syn_lqi.n_bytes_nn_bias);

    //printf("arm coded w%ld b%ld bytes w:%02x %02x %02x\n", cc.m_arm_weights_hevc.size(), cc.m_arm_biases_hevc.size(), cc.m_arm_weights_hevc[0], cc.m_arm_weights_hevc[1], cc.m_arm_weights_hevc[2]);
    //printf("ups coded w%ld b%ld bytes w:%02x %02x %02x\n", cc.m_ups_weights_hevc.size(), cc.m_ups_biases_hevc.size(), cc.m_ups_weights_hevc[0], cc.m_ups_weights_hevc[1], cc.m_ups_weights_hevc[2]);
    //printf("syn coded w%ld b%ld bytes w %02x %02x %02x\n", cc.m_syn_weights_hevc.size(), cc.m_syn_biases_hevc.size(), cc.m_syn_weights_hevc[0], cc.m_syn_weights_hevc[1], cc.m_syn_weights_hevc[2]);
    //fflush(stdout);

    for (int i = 0; i < cc.n_latent_n_resolutions; i++)
        cc.m_latents_hevc.emplace_back(get_coded(f, cc.n_bytes_per_latent[i]));

    return true;
}

struct cc_bs_frame *cc_bs::decode_frame(struct cc_bs_frame const *prev_coded_I_frame_symbols, struct cc_bs_frame const *prev_coded_frame_symbols, int verbosity)
{
    cc_bs_frame *result = new cc_bs_frame();
    if (!read_frame_header(m_f, result->m_frame_header, verbosity))
    {
        delete result;
        return NULL;
    }

    // Do we use the previous coded I or the previous coded?
    // we do not use the previous coded if it is before the previous coded I, and the current frame is after the previous coded I.
    struct cc_bs_frame const *prev_frame_symbols;
    if (prev_coded_I_frame_symbols == NULL)
        prev_frame_symbols = NULL;
    else if (prev_coded_frame_symbols == NULL)
        prev_frame_symbols = prev_coded_I_frame_symbols;
    else
    {
        if (result->m_frame_header.display_index > prev_coded_I_frame_symbols->m_frame_header.display_index
            && prev_coded_frame_symbols->m_frame_header.display_index < prev_coded_I_frame_symbols->m_frame_header.display_index)
            prev_frame_symbols = prev_coded_I_frame_symbols; // A P after the I at EOS.
        else
            prev_frame_symbols = prev_coded_frame_symbols; // the normal case.
    }

    for (int i = 0; i < (result->m_frame_header.frame_type == 'I' ? 1 : 2); i++)
    {
        result->m_coolchics.emplace_back();
        if (result->m_frame_header.topology_copy[i] != 0)
            result->m_coolchics[result->m_coolchics.size()-1].copy_archi_from(
                        prev_frame_symbols->m_coolchics[prev_frame_symbols->m_frame_header.frame_type != 'I' ? i : 0],
                        result->m_frame_header.topology_copy[i]);
        if (!read_cc_archi(result->m_coolchics[result->m_coolchics.size()-1], result->m_frame_header.topology_copy[i], m_f))
        {
            delete result;
            return NULL;
        }
        if (!read_cc_lengths(result->m_coolchics[result->m_coolchics.size()-1], result->m_frame_header.latents_zero[i], m_f))
        {
            delete result;
            return NULL;
        }
    }

    if (verbosity >= 2)
    {
        printf("FRAME HEADER\n");
        printf("    n_bytes_header: %d\n", result->m_frame_header.n_bytes_header);
        printf("    display_index: %d\n", result->m_frame_header.display_index);
        printf("    frame_type: %c\n", result->m_frame_header.frame_type);
        printf("    topology_copy [0] 0x%x [1] 0x%x from %d\n",
                        result->m_frame_header.topology_copy[0], result->m_frame_header.topology_copy[1],
                        prev_frame_symbols == NULL ? -1 : prev_frame_symbols->m_frame_header.display_index);
        printf("    latents_zero: %d %d\n", result->m_frame_header.latents_zero[0], result->m_frame_header.latents_zero[1]);
        printf("    global_flow_1: %d,%d\n", result->m_frame_header.global_flow[0][0], result->m_frame_header.global_flow[0][1]);
        printf("    global_flow_2: %d,%d\n", result->m_frame_header.global_flow[1][0], result->m_frame_header.global_flow[1][1]);

        for (auto &cc: result->m_coolchics)
            cc.print();
    }

    // read coolchic weights and latents.
    for (auto &cc: result->m_coolchics)
    {
        if (!read_cc_content(cc, m_f))
        {
            delete result;
            return NULL;
        }
    }

    return result;
}
