/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


/*
 * bitstream reading.
 */
// !!! enum FRAME_DATA_TYPE
struct cc_bs_gop_header
{
    int n_bytes_header;
    int img_h;
    int img_w;
    int frame_data_type; // 0 rgb, 1 yuv420, 2 yuv444 (unsupported)
    int bitdepth;
    int intra_period;
    int p_period;
};

struct cc_bs_syn_layer {
    int n_out_ft;
    int ks;
    bool residual;
    bool relu;
};

// quantization information for module weights or biases.  Possibly multiple layers encoded.
struct cc_bs_layer_quant_info
{
    int q_step_index_nn_weight;
    int q_step_index_nn_bias;
    int scale_index_nn_weight;
    int scale_index_nn_bias;
    int n_bytes_nn_weight;
    int n_bytes_nn_bias;
};

struct cc_bs_frame_header
{
    int n_bytes_header;
    int n_latent_n_resolutions;
    int latent_n_2d_grid;
    std::vector<int> n_bytes_per_latent;
    std::vector<int> n_ft_per_latent;
    int n_hidden_layers_arm;
    int dim_arm;
    int upsampling_kern_size;
    int static_upsampling_kernel;
    int flow_gain;
    int n_syn_layers;
    std::vector<struct cc_bs_syn_layer> layers_synthesis;
    struct cc_bs_layer_quant_info arm_lqi;
    struct cc_bs_layer_quant_info ups_lqi;
    struct cc_bs_layer_quant_info syn_lqi;

    int ac_max_val_nn;
    int ac_max_val_latent;
    int display_index;

    int hls_sig_blksize;
};

//bool read_gop_header(FILE *bitstream, struct bitstream_gop_header &gop_header);
//bool read_frame_header(FILE *bitstream, struct bitstream_frame_header &frame_header);

struct cc_bs_frame {
    struct cc_bs_frame_header m_frame_header;
    std::vector<unsigned char> m_arm_weights_hevc;
    std::vector<unsigned char> m_arm_biases_hevc;
    std::vector<unsigned char> m_ups_weights_hevc;
    std::vector<unsigned char> m_syn_weights_hevc;
    std::vector<unsigned char> m_syn_biases_hevc;
    std::vector<std::vector<unsigned char>> m_latents_hevc;
};

class cc_bs {
public:
    cc_bs() { m_f = NULL; }
    ~cc_bs() { if (m_f != NULL) fclose(m_f); }
    bool open(std::string filename);
    struct cc_bs_frame *decode_frame();
private:
    bool read_gop_header();

public:
    std::string m_filename;
    struct cc_bs_gop_header m_gop_header;
private:
    FILE *m_f;
};
