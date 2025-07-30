
#ifdef CCDECAPI_AVX2_OPTIONAL
#define CCDECAPI_AVX2
#endif

#include "common.h"
#include "TDecBinCoderCABAC.h"
#include "cc-contexts.h"
#include "cc-bac.h"
#include "cc-bitstream.h"
#include "cc-frame-decoder.h"

#ifdef CCDECAPI_AVX2
#include "arm_avx2.h"
#include "ups_avx2.h"
#include "syn_avx2.h"
#endif

#include "arm_cpu.h"
#include "ups_cpu.h"
#include "syn_cpu.h"

extern float time_arm_seconds;
extern float time_ups_seconds;
extern float time_syn_seconds;
extern float time_blend_seconds;

int const Q_STEP_ARM_WEIGHT_SHIFT[] = {
    8, // 1.0/(1<<8),
    7, // 1.0/(1<<7),
    6, // 1.0/(1<<6),
    5, // 1.0/(1<<5),
    4, // 1.0/(1<<4),
    3, // 1.0/(1<<3),
    2, // 1.0/(1<<2),
    1, // 1.0/(1<<1),
    0 // 1.0/(1<<0),
    };

int const Q_STEP_ARM_BIAS_SHIFT[] = {
    16, // 1.0/(1<<16),
    15, // 1.0/(1<<15),
    14, // 1.0/(1<<14),
    13, // 1.0/(1<<13),
    12, // 1.0/(1<<12),
    11, // 1.0/(1<<11),
    10, // 1.0/(1<<10),
    9,  // 1.0/(1<<9),
    8,  // 1.0/(1<<8),
    7,  // 1.0/(1<<7),
    6,  // 1.0/(1<<6),
    5,  // 1.0/(1<<5),
    4,  // 1.0/(1<<4),
    3,  // 1.0/(1<<3),
    2,  // 1.0/(1<<2),
    1,  // 1.0/(1<<1),
    0   // 1.0/(1<<0),
    };

int const Q_STEP_UPS_SHIFT[] = {
    // 16, // 1.0/(1<<16),
    // 15, // 1.0/(1<<15),
    // 14, // 1.0/(1<<14),
    // 13, // 1.0/(1<<13),
    12, // 1.0/(1<<12),
    11, // 1.0/(1<<11),
    10, // 1.0/(1<<10),
     9, // 1.0/(1<<9),
     8, // 1.0/(1<<8),
     7, // 1.0/(1<<7),
     6, // 1.0/(1<<6),
     5, // 1.0/(1<<5),
     4, // 1.0/(1<<4),
     3, // 1.0/(1<<3),
     2, // 1.0/(1<<2),
     1, // 1.0/(1<<1),
     0, // 1.0/(1<<0),
    };

int const *const Q_STEP_SYN_WEIGHT_SHIFT = Q_STEP_UPS_SHIFT;

int const Q_STEP_SYN_BIAS_SHIFT[] = {
    24, // 1.0/(1<<24),
    23, // 1.0/(1<<23),
    22, // 1.0/(1<<22),
    21, // 1.0/(1<<21),
    20, // 1.0/(1<<20),
    19, // 1.0/(1<<19),
    18, // 1.0/(1<<18),
    17, // 1.0/(1<<17),
    16, // 1.0/(1<<16),
    15, // 1.0/(1<<15),
    14, // 1.0/(1<<14),
    13, // 1.0/(1<<13),
    12, // 1.0/(1<<12),
    11, // 1.0/(1<<11),
    10, // 1.0/(1<<10),
    9,  // 1.0/(1<<9),
    8,  // 1.0/(1<<8),
    7,  // 1.0/(1<<7),
    6,  // 1.0/(1<<6),
    5,  // 1.0/(1<<5),
    4,  // 1.0/(1<<4),
    3,  // 1.0/(1<<3),
    2,  // 1.0/(1<<2),
    1,  // 1.0/(1<<1),
    0   // 1.0/(1<<0),
    };

// Return a table of n_contexts values indicating pixel offsets from current to context pixels.
int32_t *get_context_indicies(int n_contexts, int stride)
{
    static int32_t indicies_constant[32];
    if (n_contexts == 8)
    {
        int32_t indicies[] = {                                       -3*stride+0,
                                                                     -2*stride+0,
                                                        -1*stride-1, -1*stride+0, -1*stride+1,
                              -0*stride-3, -0*stride-2, -0*stride-1 };
        memcpy(indicies_constant, indicies, n_contexts*sizeof(indicies[0]));
    }
    else if (n_contexts == 16)
    {
        int32_t indicies[] = {                                       -3*stride+0, -3*stride+1,
                                           -2*stride-2, -2*stride-1, -2*stride+0, -2*stride+1, -2*stride+2,
                              -1*stride-3, -1*stride-2, -1*stride-1, -1*stride+0, -1*stride+1, -1*stride+2,
                              -0*stride-3, -0*stride-2, -0*stride-1 };
        memcpy(indicies_constant, indicies, n_contexts*sizeof(indicies[0]));
    }
    else if (n_contexts == 24)
    {
        int32_t indicies[] = {                                                    -4*stride+0,
                                                        -3*stride-2, -3*stride-1, -3*stride+0, -3*stride+1, -3*stride+2,
                                           -2*stride-3, -2*stride-2, -2*stride-1, -2*stride+0, -2*stride+1, -2*stride+2, -2*stride+3,
                                           -1*stride-3, -1*stride-2, -1*stride-1, -1*stride+0, -1*stride+1, -1*stride+2, -1*stride+3,
                              -0*stride-4, -0*stride-3, -0*stride-2, -0*stride-1 };
        memcpy(indicies_constant, indicies, n_contexts*sizeof(indicies[0]));
    }
    else if (n_contexts == 32)
    {
        int32_t indicies[] = {                          -4*stride-2, -4*stride-1, -4*stride+0, -4*stride+1,
                                           -3*stride-3, -3*stride-2, -3*stride-1, -3*stride+0, -3*stride+1, -3*stride+2, -3*stride+3,
                                           -2*stride-3, -2*stride-2, -2*stride-1, -2*stride+0, -2*stride+1, -2*stride+2, -2*stride+3, -2*stride+4,
                              -1*stride-4, -1*stride-3, -1*stride-2, -1*stride-1, -1*stride+0, -1*stride+1, -1*stride+2, -1*stride+3, -1*stride+4,
                              -0*stride-4, -0*stride-3, -0*stride-2, -0*stride-1 };
        memcpy(indicies_constant, indicies, n_contexts*sizeof(indicies[0]));
    }
    else
    {
        printf("n_context_pixels: can only be 8, 16, 24, 32, not %d\n", n_contexts);
        exit(1);
    }
    return indicies_constant;
}

// decode with dequant, return with precision.
void decode_weights_qi(int32_t *result, TDecBinCABAC *cabac, int count, int n_weights, int q_step_shift, int precision)
{
    for (int i = 0; i < n_weights; i++)
    {
        int tmp = cabac->decodeExGolomb(count);
        if (tmp != 0)
        {
            if (cabac->decodeBinEP() != 0)
                tmp = -tmp;
        }

        // want val << precision >> q_step_shift
        if (precision > q_step_shift)
            tmp <<= (precision-q_step_shift);
        else if (precision < q_step_shift)
        {
            printf("decoding weights: qstepshift %d > precision %d\n", q_step_shift, precision);
            exit(1);
        }

        result[i] = tmp;
    }
}

void decode_weights_qi(weights_biases &result, TDecBinCABAC *cabac, int count, int n_weights, int q_step_shift, int precision)
{
    result.update_to(n_weights);
    decode_weights_qi(result.data, cabac, count, n_weights, q_step_shift, precision);
}

// !!! temporary -- float synthesis weights/biases
void decode_weights_qi(float *result, TDecBinCABAC *cabac, int count, int n_weights, int q_step_shift, int precision)
{
    for (int i = 0; i < n_weights; i++)
    {
        int tmp = cabac->decodeExGolomb(count);
        if (tmp != 0)
        {
            if (cabac->decodeBinEP() != 0)
                tmp = -tmp;
        }

        result[i] = tmp/((1<<q_step_shift)+0.0);
    }
}

void decode_weights_qi(buffer<float> &result, TDecBinCABAC *cabac, int count, int n_weights, int q_step_shift, int precision)
{
    result.update_to(n_weights);
    decode_weights_qi(result.data, cabac, count, n_weights, q_step_shift, precision);
}

// We take a kernel size ks, but only read (ks+1)/2 weights.  These weights are mirrored to produce the full kernel.
void decode_upsweights_qi(weights_biases_ups &result, TDecBinCABAC *cabac, int count, int ks, int q_step_shift, int precision)
{
    int nw = (ks+1)/2; // number of weights to read.
    result.update_to(ks); // we allocate to allow mirroring.
    decode_weights_qi(result.data, cabac, count, nw, q_step_shift, precision); // read the 1st half.
    // mirror last half
    for (int i = 0; i < nw/2*2; i++)
    {
        result.data[ks-1-i] = result.data[i];
    }
    fflush(stdout);
}

void cc_frame_decoder::read_arm(struct cc_bs_frame_coolchic &frame_symbols)
{
    if (frame_symbols.n_hidden_layers_arm != m_mlp_n_hidden_layers_arm)
    {
        m_mlp_n_hidden_layers_arm = frame_symbols.n_hidden_layers_arm;
        delete [] m_mlpw_t;
        delete [] m_mlpb;
        m_mlpw_t = new weights_biases[m_mlp_n_hidden_layers_arm];
        m_mlpb = new weights_biases[m_mlp_n_hidden_layers_arm];
    }

    InputBitstream bs_weights;
    std::vector<UChar> &bs_fifo_weights = bs_weights.getFifo();
    InputBitstream bs_biases;
    std::vector<UChar> &bs_fifo_biases = bs_biases.getFifo();

    TDecBinCABAC cabac_weights;
    TDecBinCABAC cabac_biases;

    bs_fifo_weights = frame_symbols.m_arm_weights_hevc;
    cabac_weights.init(&bs_weights);
    cabac_weights.start();

    bs_fifo_biases = frame_symbols.m_arm_biases_hevc;
    cabac_biases.init(&bs_biases);
    cabac_biases.start();

    struct cc_bs_layer_quant_info &arm_lqi = frame_symbols.arm_lqi;

    int q_step_w_shift = Q_STEP_ARM_WEIGHT_SHIFT[arm_lqi.q_step_index_nn_weight];
    int q_step_b_shift = Q_STEP_ARM_BIAS_SHIFT[arm_lqi.q_step_index_nn_bias];
    int arm_n_features = frame_symbols.dim_arm;
    int32_t bucket[arm_n_features*arm_n_features]; // we transpose from this.

    for (int idx = 0; idx < frame_symbols.n_hidden_layers_arm; idx++)
    {
        decode_weights_qi(&bucket[0], &cabac_weights, arm_lqi.scale_index_nn_weight, arm_n_features*arm_n_features, q_step_w_shift, ARM_PRECISION);
        // transpose.
        m_mlpw_t[idx].update_to(arm_n_features*arm_n_features);
        int kidx = 0;
        for (int ky = 0; ky < arm_n_features; ky++)
            for (int kx = 0; kx < arm_n_features; kx++)
            {
                m_mlpw_t[idx].data[kidx++] = bucket[kx*arm_n_features+ky];
            }
        decode_weights_qi(m_mlpb[idx], &cabac_biases, arm_lqi.scale_index_nn_bias, arm_n_features, q_step_b_shift, ARM_PRECISION*2);
    }
    decode_weights_qi(m_mlpwOUT, &cabac_weights, arm_lqi.scale_index_nn_weight, arm_n_features*2, q_step_w_shift, ARM_PRECISION);
    decode_weights_qi(m_mlpbOUT, &cabac_biases,  arm_lqi.scale_index_nn_bias, 2, q_step_b_shift, ARM_PRECISION*2);
}


void cc_frame_decoder::read_ups(struct cc_bs_frame_coolchic &frame_symbols)
{
    // zero latents => no upsampling.
    if (frame_symbols.latents_zero)
        return;

    int n_ups = frame_symbols.n_ups_kernel;
    int n_ups_preconcat = frame_symbols.n_ups_preconcat_kernel;
    if (n_ups != m_ups_n)
    {
        delete[] m_upsw;
        m_upsw = new weights_biases_ups[n_ups];
        m_ups_n = n_ups;
    }
    if (n_ups_preconcat != m_ups_n_preconcat)
    {
        delete[] m_upsw_preconcat;
        m_upsw_preconcat = new weights_biases_ups[n_ups_preconcat];
        m_ups_n_preconcat = n_ups_preconcat;
    }

    InputBitstream bs_weights;
    std::vector<UChar> &bs_fifo_weights = bs_weights.getFifo();

    TDecBinCABAC cabac_weights;

    bs_fifo_weights = frame_symbols.m_ups_weights_hevc;
    cabac_weights.init(&bs_weights);
    cabac_weights.start();
    struct cc_bs_layer_quant_info &ups_lqi = frame_symbols.ups_lqi;

    int q_step_w_shift = Q_STEP_UPS_SHIFT[ups_lqi.q_step_index_nn_weight];

    // read ups layers
    for (int lidx = 0; lidx < m_ups_n; lidx++)
    {
        decode_upsweights_qi(m_upsw[lidx], &cabac_weights, ups_lqi.scale_index_nn_weight, frame_symbols.ups_k_size, q_step_w_shift, UPS_PRECISION);
    }
    // read ups_preconcat layers.
    for (int lidx = 0; lidx < m_ups_n_preconcat; lidx++)
    {
        decode_upsweights_qi(m_upsw_preconcat[lidx], &cabac_weights, ups_lqi.scale_index_nn_weight, frame_symbols.ups_preconcat_k_size, q_step_w_shift, UPS_PRECISION);
    }
}

void cc_frame_decoder::read_syn(struct cc_bs_frame_coolchic &frame_symbols)
{
    if (frame_symbols.n_syn_layers != m_syn_n_layers)
    {
        delete [] m_synw;
        delete [] m_synb;
        m_syn_n_layers = frame_symbols.n_syn_layers;
        m_syn_blends.update_to(m_syn_n_layers);
        m_synw = new weights_biases_syn[m_syn_n_branches*m_syn_n_layers];
        m_synb = new weights_biases_syn[m_syn_n_branches*m_syn_n_layers];
    }

    InputBitstream bs_weights;
    std::vector<UChar> &bs_fifo_weights = bs_weights.getFifo();
    InputBitstream bs_biases;
    std::vector<UChar> &bs_fifo_biases = bs_biases.getFifo();

    TDecBinCABAC cabac_weights;
    TDecBinCABAC cabac_biases;

    bs_fifo_weights = frame_symbols.m_syn_weights_hevc;
    bs_fifo_biases = frame_symbols.m_syn_biases_hevc;
    cabac_weights.init(&bs_weights);
    cabac_weights.start();
    cabac_biases.init(&bs_biases);
    cabac_biases.start();
    struct cc_bs_layer_quant_info &syn_lqi = frame_symbols.syn_lqi;

    int q_step_w_shift = Q_STEP_SYN_WEIGHT_SHIFT[syn_lqi.q_step_index_nn_weight];
    int q_step_b_shift = Q_STEP_SYN_BIAS_SHIFT[syn_lqi.q_step_index_nn_bias];

    // blend values.
    if (m_syn_n_branches > 1)
    {
        decode_weights_qi(m_syn_blends, &cabac_weights, syn_lqi.scale_index_nn_weight, m_syn_n_branches, q_step_w_shift, SYN_WEIGHT_PRECISION);
    }

    // layer weights.
    for (int bidx = 0; bidx < 1; bidx++) // no more branches.
    {
        int n_in_ft = frame_symbols.latent_n_2d_grid;
        for (int lidx = 0; lidx < frame_symbols.n_syn_layers; lidx++)
        {
            struct cc_bs_syn_layer &syn = frame_symbols.layers_synthesis[lidx];
            int n_weights = n_in_ft * syn.ks*syn.ks * syn.n_out_ft;
            int n_biases = syn.n_out_ft;

            decode_weights_qi(m_synw[get_syn_idx(bidx, lidx)], &cabac_weights, syn_lqi.scale_index_nn_weight, n_weights, q_step_w_shift, SYN_WEIGHT_PRECISION);
            decode_weights_qi(m_synb[get_syn_idx(bidx, lidx)], &cabac_biases,  syn_lqi.scale_index_nn_bias, n_biases,  q_step_b_shift, SYN_WEIGHT_PRECISION*2);

            n_in_ft = syn.n_out_ft;
        }
    }
}

// check to see if the leading couple of syns are compatible with being fused
// together.
// Memory increase avoidance is so important we use cpu-mode in avx2 if it's
// not otherwise optimized.
bool cc_frame_decoder::can_fuse(struct cc_bs_frame_coolchic &frame_symbols)
{
    auto &synthesis = frame_symbols.layers_synthesis;

    bool fused = synthesis[0].ks == 1 && synthesis[1].ks == 1;
    return fused;
}

// Make sure buffers are correctly sized.
// We can now target a downsampled result (ie, motion with 0-feature layers)
void cc_frame_decoder::check_allocations(struct cc_bs_frame_coolchic &frame_symbols)
{
    int const latent_n_resolutions = frame_symbols.latent_n_2d_grid;

    int n_max_planes = latent_n_resolutions;
    m_arm_pad = 4; // by how much the arm frame is padded.
    m_ups_pad = (std::max(frame_symbols.ups_k_size, frame_symbols.ups_preconcat_k_size)+1)/2;

    int syn_max_ks = 1;
    for (int syn_idx = 0; syn_idx < frame_symbols.n_syn_layers; syn_idx++)
    {
        if (syn_idx == 0 && can_fuse(frame_symbols))
            continue; // ignore hidden layer size on fused layer.
        int n_syn_out = frame_symbols.layers_synthesis[syn_idx].n_out_ft;
        if (n_syn_out > n_max_planes)
            n_max_planes = n_syn_out;
        int n_pad = frame_symbols.layers_synthesis[syn_idx].ks/2;
        if (n_pad > m_max_pad)
            m_max_pad = n_pad;
        syn_max_ks = std::max(syn_max_ks, frame_symbols.layers_synthesis[syn_idx].ks);
    }

    int syn_pad = (syn_max_ks+1)/2;
    m_max_pad = std::max(std::max(m_arm_pad, m_ups_pad), syn_pad); // during arm, ups and syn.

    if (m_verbosity >= 3)
        printf("MAX PLANES SET TO %d; MAX PAD SET TO %d\n", n_max_planes, m_max_pad);

    m_target_img_h = m_gop_header.img_h/(1<<frame_symbols.n_leading_zero_feature_layers);
    m_target_img_w = m_gop_header.img_w/(1<<frame_symbols.n_leading_zero_feature_layers);

    // !!! allocate all for the moment for bisyn.
    m_pre_syn_1.update_to(m_target_img_h, m_gop_header.img_w, m_max_pad, n_max_planes); // !!! temporary !!! only used if SYN_INT_FLOAT is float.
    m_syn_1.update_to(m_target_img_h, m_target_img_w, m_max_pad, n_max_planes);
    m_syn_2.update_to(m_target_img_h, m_target_img_w, m_max_pad, n_max_planes);
    m_syn_3.update_to(m_target_img_h, m_target_img_w, m_max_pad, n_max_planes);
    m_syn_tmp.update_to(m_target_img_h, m_target_img_w, m_max_pad, n_max_planes);

    // pyramid sizes.
    m_zero_layer.resize(frame_symbols.n_latent_n_resolutions);
    m_h_pyramid.resize(frame_symbols.n_latent_n_resolutions);
    m_w_pyramid.resize(frame_symbols.n_latent_n_resolutions);

    // pyramid storage -- enough pad for arm and ups.  includes highest res layer.
    m_plane_pyramid.resize(frame_symbols.n_latent_n_resolutions);
    // !!! ups float temporary
    m_plane_pyramid_for_ups.resize(frame_symbols.n_latent_n_resolutions);

    for (int layer_number = 0, h_grid = m_gop_header.img_h, w_grid = m_gop_header.img_w;
         layer_number < frame_symbols.n_latent_n_resolutions;
         layer_number++, h_grid = (h_grid+1)/2, w_grid = (w_grid+1)/2)
    {
        m_h_pyramid[layer_number] = h_grid;
        m_w_pyramid[layer_number] = w_grid;
        m_plane_pyramid[layer_number].update_to(h_grid, w_grid, m_max_pad, 1);
        // !!! ups float temporary
        m_plane_pyramid_for_ups[layer_number].update_to(h_grid, w_grid, m_max_pad, 1);
    }

    // We need a couple of extra buffers for pyramid upsampling and refinement.  It's not in-place.
    // 1/4 sized for internal refinement target.  (Full-sized goes directly to syn-land.)
    m_ups_h2w2.update_to(m_h_pyramid[1], m_w_pyramid[1], m_ups_pad, 1);
    // full-height, full-width for actual upsample.
    m_ups_hw.update_to(m_h_pyramid[0], m_w_pyramid[0], m_ups_pad, 1);
}

void ups_refine_cpu(int ks_param, UPS_INT_FLOAT *kw, frame_memory<UPS_INT_FLOAT> &in, frame_memory<UPS_INT_FLOAT> &out, int ups_src_precision, frame_memory<UPS_INT_FLOAT> &tmp)
{
    if (ks_param == 7)
        ups_refine_ks7_cpu(ks_param, kw, in, out, ups_src_precision, tmp);
    else
        ups_refine_ksX_cpu(ks_param, kw, in, out, ups_src_precision, tmp);
}

void ups_upsample_cpu(int ksx2, UPS_INT_FLOAT *kw, frame_memory<UPS_INT_FLOAT> &in, frame_memory<UPS_INT_FLOAT> &out, int out_plane, int ups_src_precision, frame_memory<UPS_INT_FLOAT> &tmp)
{
    if (ksx2 == 8)
        ups_upsample_ks8_cpu(ksx2, kw, in, out, out_plane, ups_src_precision, tmp);
    else
        ups_upsample_ksX_cpu(ksx2, kw, in, out, out_plane, ups_src_precision, tmp);
}

#ifdef CCDECAPI_AVX2
void ups_refine_avx2(int ks_param, UPS_INT_FLOAT *kw, frame_memory<UPS_INT_FLOAT> &in, frame_memory<UPS_INT_FLOAT> &out, int ups_src_precision, frame_memory<UPS_INT_FLOAT> &tmp)
{
    if (ks_param == 7)
        ups_refine_ks7_avx2(ks_param, kw, in, out, ups_src_precision, tmp);
    else
        ups_refine_ksX_avx2(ks_param, kw, in, out, ups_src_precision, tmp);
}

void ups_upsample_avx2(int ksx2, UPS_INT_FLOAT *kw, frame_memory<UPS_INT_FLOAT> &in, frame_memory<UPS_INT_FLOAT> &out, int out_plane, int ups_src_precision, frame_memory<UPS_INT_FLOAT> &tmp)
{
    if (ksx2 == 8 && ups_src_precision == ARM_PRECISION)
        ups_upsample_ks8_ARMPREC_avx2(ksx2, kw, in, out, out_plane, ups_src_precision, tmp);
    else if (ksx2 == 8 && ups_src_precision == UPS_PRECISION)
        ups_upsample_ks8_UPSPREC_avx2(ksx2, kw, in, out, out_plane, ups_src_precision, tmp);
    else
        ups_upsample_ksX_avx2(ksx2, kw, in, out, out_plane, ups_src_precision, tmp);
}
#endif

void cc_frame_decoder::run_arm(struct cc_bs_frame_coolchic &frame_symbols)
{
    // RUN ARM
    // latent-layer processing.
    for (int layer_number = 0; layer_number < frame_symbols.n_latent_n_resolutions; layer_number++)
    {
        int const h_grid = m_h_pyramid[layer_number];
        int const w_grid = m_w_pyramid[layer_number];

        if (m_verbosity >= 3)
            printf("arm:starting layer %d\n", layer_number);
        fflush(stdout);

        auto &bytes = frame_symbols.m_latents_hevc[layer_number];
        if (bytes.size() == 0)
        {
            // produce a zero layer.
            m_zero_layer[layer_number] = true;
            continue;
        }
        m_zero_layer[layer_number] = false;

        //frame_memory *dest = layer_number == 0 ? &m_syn_1 : &m_plane_pyramid[layer_number];
        frame_memory<int32_t> *dest = &m_plane_pyramid[layer_number];
        dest->zero_pad(0, m_arm_pad);

        // BAC decoding:
        InputBitstream bsBAC;
        std::vector<UChar> &bs = bsBAC.getFifo();
        bs = bytes;

        TDecBinCABAC layerBAC;
        layerBAC.init(&bsBAC);
        layerBAC.start();

        BACContext bac_context;
        bac_context.set_layer(&layerBAC, h_grid, w_grid, frame_symbols.hls_sig_blksize);

        const auto time_arm_start = std::chrono::steady_clock::now();

#ifdef CCDECAPI_AVX2
        if (m_use_avx2 && frame_symbols.dim_arm == 8)
            custom_conv_11_int32_avx2_8_X_X(
                                      m_mlpw_t, m_mlpb,
                                      &m_mlpwOUT, &m_mlpbOUT,
                                      get_context_indicies(frame_symbols.dim_arm, dest->stride), frame_symbols.dim_arm, frame_symbols.n_hidden_layers_arm,
                                      dest->origin(),
                                      h_grid, w_grid, (dest->stride-w_grid)/2,
                                      bac_context
                                      );
        else if (m_use_avx2 && frame_symbols.dim_arm == 16)
            custom_conv_11_int32_avx2_16_X_X(
                                      m_mlpw_t, m_mlpb,
                                      &m_mlpwOUT, &m_mlpbOUT,
                                      get_context_indicies(frame_symbols.dim_arm, dest->stride), frame_symbols.dim_arm, frame_symbols.n_hidden_layers_arm,
                                      dest->origin(),
                                      h_grid, w_grid, (dest->stride-w_grid)/2,
                                      bac_context
                                      );
        else if (m_use_avx2 && frame_symbols.dim_arm == 24)
            custom_conv_11_int32_avx2_24_X_X(
                                      m_mlpw_t, m_mlpb,
                                      &m_mlpwOUT, &m_mlpbOUT,
                                      get_context_indicies(frame_symbols.dim_arm, dest->stride), frame_symbols.dim_arm, frame_symbols.n_hidden_layers_arm,
                                      dest->origin(),
                                      h_grid, w_grid, (dest->stride-w_grid)/2,
                                      bac_context
                                      );
        else if (m_use_avx2 && frame_symbols.dim_arm == 32)
            custom_conv_11_int32_avx2_32_X_X(
                                      m_mlpw_t, m_mlpb,
                                      &m_mlpwOUT, &m_mlpbOUT,
                                      get_context_indicies(frame_symbols.dim_arm, dest->stride), frame_symbols.dim_arm, frame_symbols.n_hidden_layers_arm,
                                      dest->origin(),
                                      h_grid, w_grid, (dest->stride-w_grid)/2,
                                      bac_context
                                      );
        else
#endif
            custom_conv_11_int32_cpu_X_X_X(
                                      m_mlpw_t, m_mlpb,
                                      &m_mlpwOUT, &m_mlpbOUT,
                                      get_context_indicies(frame_symbols.dim_arm, dest->stride), frame_symbols.dim_arm, frame_symbols.n_hidden_layers_arm,

                                      dest->origin(),
                                      h_grid, w_grid, (dest->stride-w_grid)/2,

                                      bac_context
                                      );

        const auto time_arm_done = std::chrono::steady_clock::now();
        const std::chrono::duration<double> arm_elapsed = (time_arm_done-time_arm_start);
        time_arm_seconds += (float)arm_elapsed.count();

    } // layer.

    if (m_verbosity >= 100)
    {
        printf("ARM OUTPUTS\n");
        for (int p = 0; p < frame_symbols.n_latent_n_resolutions; p++)
        {
            printf("ARMPYRAMID %d ", p);
            m_plane_pyramid[p].print_start(0, "ARM", -1, ARM_PRECISION);
        }
    }
}

frame_memory<int32_t> *hoop(frame_memory<int32_t> *syn_1, frame_memory<int32_t> *pre_syn_1)
{
    return syn_1;
}
frame_memory<float> *hoop(frame_memory<float> *syn_1, frame_memory<float> *pre_syn_1)
{
    return syn_1;
}
frame_memory<int32_t> *hoop(frame_memory<float> *syn_1, frame_memory<int32_t> *pre_syn_1)
{
    return pre_syn_1;
}

// the input pyramid for ups: float if UPS_INT_FLOAT is float.
std::vector<frame_memory<float>> *hoop2(std::vector<frame_memory<int32_t>> *plane_pyramid, std::vector<frame_memory<float>> *plane_pyramid_for_ups)
{
    return plane_pyramid_for_ups;
}

// the input pyramid for ups: int32_T if UPS_INT_FLOAT is int32_t.
std::vector<frame_memory<int32_t>> *hoop2(std::vector<frame_memory<int32_t>> *plane_pyramid, std::vector<frame_memory<int32_t>> *plane_pyramid_for_ups)
{
    return plane_pyramid;
}

void cc_frame_decoder::run_ups(struct cc_bs_frame_coolchic &frame_symbols)
{
    const auto time_ups_start = std::chrono::steady_clock::now();

    // zero latents => no ups.
    // we set out output to zero.
    if (frame_symbols.latents_zero)
    {
        for (int p = 0; p < m_syn_1.planes; p++)
            m_syn_1.zero_plane_content(p);

        const auto time_ups_end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> hand_ups_elapsed_seconds = time_ups_end - time_ups_start;
        time_ups_seconds += hand_ups_elapsed_seconds.count();

        return;
    }

    std::vector<frame_memory<UPS_INT_FLOAT>> *ups_input_pyramid;
    if constexpr(std::is_same<UPS_INT_FLOAT, float>::value)
    {
        // !!! temporary -- convert the int arm output to float ups input.
        for (int layer_number = 0; layer_number < frame_symbols.n_latent_n_resolutions; layer_number++)
        {
            if (frame_symbols.n_ft_per_latent[layer_number] == 0)
                continue;
            for (int y = 0; y < m_plane_pyramid[layer_number].h; y++)
                for (int x = 0; x < m_plane_pyramid[layer_number].w; x++)
                    m_plane_pyramid_for_ups[layer_number].plane_pixel(0, y, x)[0] = m_plane_pyramid[layer_number].plane_pixel(0, y, x)[0]/((1<<ARM_PRECISION)+0.0);
        }
        if (m_verbosity >= 100)
            printf("converted ARM output to float for UPS\n");
        ups_input_pyramid = hoop2(&m_plane_pyramid, &m_plane_pyramid_for_ups);
    }
    else
    {
        ups_input_pyramid = hoop2(&m_plane_pyramid, &m_plane_pyramid_for_ups);
    }

    // we point to a full-res integer upsample result which:
    // if SYN_INT_FLOAT is int32_t, is m_syn_1.
    // if SYN_INT_FLOAT is float, is m_pre_syn_1.  As m_syn_1 is float.
    // assign the int32_t full res option to full_res_out, preferring m_syn_1 if both are int.
    frame_memory<UPS_INT_FLOAT> *full_res_out = hoop(&m_syn_1, &m_pre_syn_1);

    // processing order full-res down to lowest-res.
    // refine & upsample each as necessary to full res, placed finally in m_syn_1.
    // the layer is not processed, and dst_plane is not incremented, if n_ft_per_latent is 0.
    // leading n_ft_per_latent == 0 layers imply a reduced resolution output -- used for motion.
    int global_dst_plane = 0;
    for (int layer_number = frame_symbols.n_leading_zero_feature_layers, h_grid = m_gop_header.img_h, w_grid = m_gop_header.img_w;
         layer_number < frame_symbols.n_latent_n_resolutions;
         layer_number++, h_grid = (h_grid+1)/2, w_grid = (w_grid+1)/2)
    {
        if (frame_symbols.n_ft_per_latent[layer_number] == 0)
        {
            continue; // skip.  we do not increment global_dst_plane.
        }

        if (m_zero_layer[layer_number])
        {
            // no need to upsample.  just zero the final content.
            full_res_out->zero_plane_content(global_dst_plane);
            // we have produced output.
            global_dst_plane += 1;
            continue;
        }

        // layer_number 0: hb_layer is (nlatents-1)%nhblayers.
        // n_resolution-2 is number of hb layers max.
        int preconcat_layer = (frame_symbols.n_latent_n_resolutions-2-layer_number)%m_ups_n_preconcat;
        if (layer_number == frame_symbols.n_leading_zero_feature_layers)
        {
            // highest res. just a refinement directly to m_syn_1, no upsampling.
#ifdef CCDECAPI_AVX2
            if (m_use_avx2)
            {
                ups_refine_avx2(frame_symbols.ups_preconcat_k_size, m_upsw_preconcat[preconcat_layer].data, (*ups_input_pyramid)[layer_number], *full_res_out, ARM_PRECISION, m_ups_hw);
            }
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL)
            else
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL) || defined(CCDECAPI_CPU)
            {
                ups_refine_cpu(frame_symbols.ups_preconcat_k_size, m_upsw_preconcat[preconcat_layer].data, (*ups_input_pyramid)[layer_number], *full_res_out, ARM_PRECISION, m_ups_hw);
            }
#endif
            // we have produced output.
            global_dst_plane += 1;
            continue;
        }

        frame_memory<UPS_INT_FLOAT> *ups_src = NULL;
        int ups_prec = 0;

        if (layer_number == frame_symbols.n_latent_n_resolutions-1)
        {
            // just upsample, no refinement.
            ups_src = &(*ups_input_pyramid)[layer_number];
            ups_prec = ARM_PRECISION;
        }
        else
        {
            // refine, then upsample.
#ifdef CCDECAPI_AVX2
            if (m_use_avx2)
            {
                ups_refine_avx2(frame_symbols.ups_preconcat_k_size, m_upsw_preconcat[preconcat_layer].data, (*ups_input_pyramid)[layer_number], m_ups_h2w2, ARM_PRECISION, m_ups_hw);
            }
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL)
            else
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL) || defined(CCDECAPI_CPU)
            {
                ups_refine_cpu(frame_symbols.ups_preconcat_k_size, m_upsw_preconcat[preconcat_layer].data, (*ups_input_pyramid)[layer_number], m_ups_h2w2, ARM_PRECISION, m_ups_hw);
            }
#endif
            ups_src = &m_ups_h2w2;
            ups_prec = UPS_PRECISION;
        }

        for (int target_layer = layer_number-1; target_layer >= frame_symbols.n_leading_zero_feature_layers; ups_src = &(*ups_input_pyramid)[target_layer], target_layer--, ups_prec = UPS_PRECISION)
        {
            // upsample layer index to use.
            int ups_layer = (frame_symbols.n_latent_n_resolutions-2-target_layer)%m_ups_n;
            // upsample, either to next pyramid level up or, instead of to [0], to m_syn_1[layer_number].
            frame_memory<UPS_INT_FLOAT> *ups_dst = NULL;
            int this_dst_plane;
            if (target_layer == frame_symbols.n_leading_zero_feature_layers)
            {
                ups_dst = full_res_out;
                this_dst_plane = global_dst_plane;
            }
            else
            {
                ups_dst = &(*ups_input_pyramid)[target_layer];
                this_dst_plane = 0;
            }
#ifdef CCDECAPI_AVX2
            if (m_use_avx2)
            {
                ups_upsample_avx2(frame_symbols.ups_k_size, m_upsw[ups_layer].data, *ups_src, *ups_dst, this_dst_plane, ups_prec, m_ups_hw);
            }
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL)
            else
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL) || defined(CCDECAPI_CPU)
            {
                ups_upsample_cpu(frame_symbols.ups_k_size, m_upsw[ups_layer].data, *ups_src, *ups_dst, this_dst_plane, ups_prec, m_ups_hw);
            }
#endif
        }

        // produced an upsampled output.
        global_dst_plane += 1;
    }

    if constexpr(std::is_same<UPS_INT_FLOAT, int32_t>::value && std::is_same<SYN_INT_FLOAT, float>::value)
    {
        printf("!!! migrating int ups out to float syn in\n");
        // !!! temporary -- migrate int32_t m_pre_syn_1 to <SYN_INT_FLOAT> m_syn_1.
        // !!! eventually use templated upsample<float> to perform post process conversion.
        for (int p = 0; p < m_syn_1.planes; p++)
        {
            for (int y = 0; y < m_syn_1.h; y++)
                for (int x = 0; x < m_syn_1.w; x++)
                    m_syn_1.plane_pixel(p, y, x)[0] = m_pre_syn_1.plane_pixel(p, y, x)[0]/((1<<UPS_PRECISION)+0.0);
        }
    }

    const auto time_ups_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> hand_ups_elapsed_seconds = time_ups_end - time_ups_start;
    time_ups_seconds += hand_ups_elapsed_seconds.count();
}

// blend an output into an accumulator.
// syn_out = syn_out + syn_in*blend[branch_no]
frame_memory<SYN_INT_FLOAT> *cc_frame_decoder::run_syn_blend1(struct cc_bs_frame_coolchic &frame_symbols, int n_planes, int branch_no, frame_memory<SYN_INT_FLOAT> *syn_in, frame_memory<SYN_INT_FLOAT> *syn_out)
{
    const auto time_blend_start = std::chrono::steady_clock::now();
#if 0
#ifdef CCDECAPI_AVX2
    if (m_use_avx2)
    {
        syn_blend1_avx2(syn_in->h, syn_in->w, syn_in->stride, syn_in->plane_stride, n_planes, syn_in->origin(), m_syn_blends.data[branch_no], syn_out->origin());
    }
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL)
    else
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL) || defined(CCDECAPI_CPU)
    {
        syn_blend1(syn_in->h, syn_in->w, syn_in->stride, syn_in->plane_stride, n_planes, syn_in->origin(), m_syn_blends.data[branch_no], syn_out->origin());
    }
#endif
#endif

    if (m_verbosity >= 100)
    {
        printf("BLEND1 INPUT\n");
        for (int p = 0; p < n_planes; p++)
        {
            printf("BLENDINPLANE %d ", p);
            syn_in->print_start(p, "BLENDIN", 10, UPS_PRECISION);
        }
    }

    syn_blend1(syn_in->h, syn_in->w, syn_in->stride, syn_in->plane_stride, n_planes, syn_in->origin(), m_syn_blends.data[branch_no], syn_out->origin());
    const auto time_blend_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> hand_blend_elapsed_seconds = time_blend_end - time_blend_start;
    time_blend_seconds += hand_blend_elapsed_seconds.count();
    return syn_out;
}

// blend two outputs to initialilze an accumulator.
// the 'in' is the 2nd syn output, the 'out' is the first syn output.
// we keep updating 'out'
// syn_out = syn_out*blend[branch_no-1] + syn_in*blend[branch_no]
frame_memory<SYN_INT_FLOAT> *cc_frame_decoder::run_syn_blend2(struct cc_bs_frame_coolchic &frame_symbols, int n_planes, int branch_no, frame_memory<SYN_INT_FLOAT> *syn_in, frame_memory<SYN_INT_FLOAT> *syn_out)
{
    const auto time_blend_start = std::chrono::steady_clock::now();
#if 0
#ifdef CCDECAPI_AVX2
    if (m_use_avx2)
    {
        syn_blend2_avx2(syn_in->h, syn_in->w, syn_in->stride, syn_in->plane_stride, n_planes, syn_in->origin(), m_syn_blends.data[branch_no], syn_out->origin(), m_syn_blends.data[branch_no-1]);
    }
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL)
    else
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL) || defined(CCDECAPI_CPU)
    {
        syn_blend2(syn_in->h, syn_in->w, syn_in->stride, syn_in->plane_stride, n_planes, syn_in->origin(), m_syn_blends.data[branch_no], syn_out->origin(), m_syn_blends.data[branch_no-1]);
    }
#endif
#endif

    if (m_verbosity >= 100)
    {
        printf("BLEND2 INPUTS\n");
        printf("1: w=%d(%f)\n", m_syn_blends.data[branch_no], (m_syn_blends.data[branch_no]/((1<<UPS_PRECISION)+0.0)));
        for (int p = 0; p < n_planes; p++)
        {
            printf("BLENDINPLANE %d ", p);
            syn_in->print_start(p, "BLENDIN1", 10, UPS_PRECISION);
        }
        printf("2: w=%d(%f)\n", m_syn_blends.data[branch_no-1], (m_syn_blends.data[branch_no]/((1<<UPS_PRECISION)+0.0)));
        for (int p = 0; p < n_planes; p++)
        {
            printf("BLENDINPLANE %d ", p);
            syn_out->print_start(p, "BLENDIN2", 10, UPS_PRECISION);
        }
    }

    syn_blend2(syn_in->h, syn_in->w, syn_in->stride, syn_in->plane_stride, n_planes, syn_in->origin(), m_syn_blends.data[branch_no], syn_out->origin(), m_syn_blends.data[branch_no-1]);
    const auto time_blend_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> hand_blend_elapsed_seconds = time_blend_end - time_blend_start;
    time_blend_seconds += hand_blend_elapsed_seconds.count();
    return syn_out;
}

// operate synth layers starting from syn_in as input.
// syn_out non-NULL implies the first syn layer will EXPLICITLY target syn_out, thereafter remaining in syn_out if possible (alternating syn_out, syn_tmp if necessary)
//     return value is either syn_out or syn_tmp.
// syn_out NULL implies syn processing will remain in syn_in as much as possible (alternating syn_in, syn_tmp if necessary)
//     return value is either syn_in or syn_tmp.
frame_memory<SYN_INT_FLOAT> *cc_frame_decoder::run_syn_branch(struct cc_bs_frame_coolchic &frame_symbols, int branch_no, frame_memory<SYN_INT_FLOAT> *syn_in, frame_memory<SYN_INT_FLOAT> *syn_out, frame_memory<SYN_INT_FLOAT> *syn_tmp)
{
    int    n_syn_in = frame_symbols.latent_n_2d_grid;
    frame_memory<SYN_INT_FLOAT> *syn_in_i = NULL; // internal in/out
    frame_memory<SYN_INT_FLOAT> *syn_out_i = NULL; // internal in/out
    if (syn_out == syn_in)
        syn_out = NULL;

    const auto time_syn_start = std::chrono::steady_clock::now();

    for (int syn_idx = 0; syn_idx < frame_symbols.n_syn_layers; syn_idx++)
    {
        auto time_this_syn_start = std::chrono::steady_clock::now();
        int n_syn_out = frame_symbols.layers_synthesis[syn_idx].n_out_ft;
        int syn_wb_idx = get_syn_idx(branch_no, syn_idx);
        if (m_verbosity >= 3)
            printf("SYN%d: k=%d #in=%d #out=%d", syn_idx, frame_symbols.layers_synthesis[syn_idx].ks, n_syn_in, n_syn_out);

        bool possible_in_place = true;
#ifdef CCDECAPI_AVX2
        if (m_use_avx2)
        {
            possible_in_place = frame_symbols.layers_synthesis[syn_idx].ks == 3
                                || (syn_idx == 0 && can_fuse(frame_symbols))
                                || (frame_symbols.layers_synthesis[syn_idx].ks == 1 && n_syn_out <= 4);
        }
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL)
        else
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL) || defined(CCDECAPI_CPU)
        {
            possible_in_place = true;
        }
#endif

        // forcing output to not in-place for first layer?
        bool in_place = possible_in_place;
        if (syn_idx == 0 && syn_out != NULL)
        {
            in_place = false; // explit targeting of non-input for output.  We will switch between syn_out, syn_tmp after as necessary.
        }

        if (syn_idx == 0)
        {
            syn_in_i = syn_in;
            syn_out_i = syn_out != NULL ? syn_out : (in_place ? syn_in_i : syn_tmp);
        }
        else
        {
            if (syn_out != NULL)
            {
                // syn_in_i is out or tmp.  determine syn_out_i
                syn_out_i = in_place ? syn_in_i : (syn_in_i == syn_out ? syn_tmp : syn_out);
            }
            else
            {
                // syn_in_i is in or tmp. determine syn_out_i
                syn_out_i = in_place ? syn_in_i : (syn_in_i == syn_in ? syn_tmp : syn_in);
            }
        }

        if (frame_symbols.layers_synthesis[syn_idx].ks > 1)
        {
            // pad syn_in, and direct output to syn_out.
            int pad = frame_symbols.layers_synthesis[syn_idx].ks/2;
            int w_padded = m_target_img_w+2*pad;
            int h_padded = m_target_img_h+2*pad;
            int origin_idx = syn_in_i->origin_idx;
            int pad_origin_idx = origin_idx-pad*syn_in_i->stride-pad;
            for (int l = 0; l < n_syn_in; l++)
                syn_in_i->custom_pad_replicate_plane_in_place_i(l, pad);
            if (m_verbosity >= 3)
                printf(" padded ");

#ifdef CCDECAPI_AVX2
            if (m_use_avx2)
            {
                if (frame_symbols.layers_synthesis[syn_idx].ks == 3)
                {
                    // optimized 3x3 with line buffer.
                    // in_place = true;
                    m_syn3x3_linebuffer.update_to(2*n_syn_out*m_target_img_w); // two lines for now, could eventually be 1 line plus a few pixels.
                    if (n_syn_in == 3 && n_syn_out == 3)
                        custom_conv_ks3_in3_out3_lb_avx2(frame_symbols.layers_synthesis[syn_idx].ks, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data,
                                                        h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, syn_out_i->origin(),
                                                        m_syn3x3_linebuffer.data,
                                                        !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (n_syn_in == 4 && n_syn_out == 4)
                        custom_conv_ks3_in4_out4_lb_avx2(frame_symbols.layers_synthesis[syn_idx].ks, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data,
                                                        h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, syn_out_i->origin(),
                                                        m_syn3x3_linebuffer.data,
                                                        !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (n_syn_in == 5 && n_syn_out == 5)
                        custom_conv_ks3_in5_out5_lb_avx2(frame_symbols.layers_synthesis[syn_idx].ks, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data,
                                                        h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, syn_out_i->origin(),
                                                        m_syn3x3_linebuffer.data,
                                                        !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (n_syn_in == 6 && n_syn_out == 6)
                        custom_conv_ks3_in6_out6_lb_avx2(frame_symbols.layers_synthesis[syn_idx].ks, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data,
                                                        h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, syn_out_i->origin(),
                                                        m_syn3x3_linebuffer.data,
                                                        !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (n_syn_in == 9 && n_syn_out == 6)
                        custom_conv_ks3_in9_out6_lb_avx2(frame_symbols.layers_synthesis[syn_idx].ks, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data,
                                                        h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, syn_out_i->origin(),
                                                        m_syn3x3_linebuffer.data,
                                                        !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (n_syn_in == 9 && n_syn_out == 9)
                        custom_conv_ks3_in9_out9_lb_avx2(frame_symbols.layers_synthesis[syn_idx].ks, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data,
                                                        h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, syn_out_i->origin(),
                                                        m_syn3x3_linebuffer.data,
                                                        !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else
                        custom_conv_ks3_inX_outX_lb_avx2(frame_symbols.layers_synthesis[syn_idx].ks, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data,
                                                        h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, syn_out_i->origin(),
                                                        m_syn3x3_linebuffer.data,
                                                        !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                }
                else
                {
                    custom_conv_ksX_inX_outX_avx2(frame_symbols.layers_synthesis[syn_idx].ks, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data,
                                                    h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, syn_out_i->origin(),
                                                    !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                }
            }
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL)
            else
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL) || defined(CCDECAPI_CPU)
            {
                if (frame_symbols.layers_synthesis[syn_idx].ks == 3)
                {
                    m_syn3x3_linebuffer.update_to(2*n_syn_out*m_target_img_w); // two lines for now, could eventually be 1 line plus a few pixels.
                    custom_conv_ks3_inX_outX_lb(frame_symbols.layers_synthesis[syn_idx].ks, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data,
                                                    h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, syn_out_i->origin(),
                                                    m_syn3x3_linebuffer.data,
                                                    !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                }
                else
                {
                    custom_conv_ksX_inX_outX(frame_symbols.layers_synthesis[syn_idx].ks, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data,
                                                    h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, syn_out_i->origin(),
                                                    !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                }
            }
#endif

            if (!in_place)
            {
                // swap our in & out.  accounting for any forced-first-time non-in-place.
                // we just update syn_in_i here, syn_out_i is calculated every loop.
                syn_in_i = syn_out_i;
            }
        }
        else
        {
            // possible in-place if ATATIME is big enough to buffer all outputs for single spatial pixel.
            // assuming max 4 here (avx2) and infinite (cpu)
            bool fused = syn_idx == 0 && can_fuse(frame_symbols);
            if (fused)
            {
                // compatible!
                int n_hidden = n_syn_out;
                n_syn_out = frame_symbols.layers_synthesis[syn_idx+1].n_out_ft;

                // transpose.
                SYN_INT_FLOAT synw_fused[n_syn_in*n_hidden];
                int kidx = 0;
                for (int kx = 0; kx < n_syn_in; kx++)
                {
                    for (int ky = 0; ky < n_hidden; ky++)
                    {
                        synw_fused[kidx++] = m_synw[syn_wb_idx].data[ky*n_syn_in+kx];
                    }
                }
#if defined(CCDECAPI_AVX2)
                if (m_use_avx2)
                {
                    if (n_syn_in == 7)
                    {
                        if (n_hidden == 48 && n_syn_out == 3)
                            custom_conv_ks1_in7_hidden48_out3_avx2(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                        else if (n_hidden == 40)
                        {
                            if (n_syn_out == 3)
                                custom_conv_ks1_in7_hidden40_out3_avx2(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                            else if (n_syn_out == 6)
                                custom_conv_ks1_in7_hidden40_out6_avx2(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                            else if (n_syn_out == 9)
                                custom_conv_ks1_in7_hidden40_out9_avx2(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                            else
                                custom_conv_ks1_inX_hiddenX_outX(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                        }
                        else if (n_hidden == 32 && n_syn_out == 3)
                            custom_conv_ks1_in7_hidden32_out3_avx2(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                        else if (n_hidden == 16)
                        {
                            if (n_syn_out == 3)
                                custom_conv_ks1_in7_hidden16_out3_avx2(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                            else if (n_syn_out == 4)
                                custom_conv_ks1_in7_hidden16_out4_avx2(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                            else if (n_syn_out == 5)
                                custom_conv_ks1_in7_hidden16_out5_avx2(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                            else
                                custom_conv_ks1_inX_hiddenX_outX(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                        }
                        else if (n_hidden == 8)
                        {
                            if (n_syn_out == 3)
                                custom_conv_ks1_in7_hidden8_out3_avx2(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                            else if (n_syn_out == 4)
                                custom_conv_ks1_in7_hidden8_out4_avx2(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                            else if (n_syn_out == 5)
                                custom_conv_ks1_in7_hidden8_out5_avx2(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                            else
                                custom_conv_ks1_inX_hiddenX_outX(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                        }
                        else
                            custom_conv_ks1_inX_hiddenX_outX(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                    }
                    else
                        custom_conv_ks1_inX_hiddenX_outX(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
                }
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL)
                else
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL) || defined(CCDECAPI_CPU)
                    custom_conv_ks1_inX_hiddenX_outX(1, synw_fused, m_synb[syn_wb_idx].data, m_synw[syn_wb_idx+1].data, m_synb[syn_wb_idx+1].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_out_i->origin());
#endif
            }
            else
            {
                // 1x1 kernel but not fused.
#ifdef CCDECAPI_AVX2
                if (m_use_avx2)
                {
                    if (n_syn_in == 7 && n_syn_out == 9)
                        custom_conv_ks1_in7_out9_avx2(1, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, syn_out_i->origin(), !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (n_syn_in == 9 && n_syn_out == 3)
                        custom_conv_ks1_in9_out3_avx2(1, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, syn_out_i->origin(), !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (n_syn_in == 9 && n_syn_out == 6)
                        custom_conv_ks1_in9_out6_avx2(1, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, syn_out_i->origin(), !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (n_syn_in == 9 && n_syn_out == 9)
                        custom_conv_ks1_in9_out9_avx2(1, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, syn_out_i->origin(), !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (n_syn_in == 7 && n_syn_out == 40)
                        custom_conv_ks1_in7_out40_avx2(1, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, syn_out_i->origin(), !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (n_syn_in == 7 && n_syn_out == 16)
                        custom_conv_ks1_in7_out16_avx2(1, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, syn_out_i->origin(), !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (                 n_syn_out == 3)
                        custom_conv_ks1_inX_out3_avx2(1, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, syn_out_i->origin(), !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (                 n_syn_out == 6)
                        custom_conv_ks1_inX_out6_avx2(1, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, syn_out_i->origin(), !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else if (                 n_syn_out == 9)
                        custom_conv_ks1_inX_out9_avx2(1, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, syn_out_i->origin(), !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                    else
                        custom_conv_ks1_inX_outX_avx2(1, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, syn_out_i->origin(), !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                }
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL)
                else
#endif
#if defined(CCDECAPI_AVX2_OPTIONAL) || defined(CCDECAPI_CPU)
                {
                    // can buffer everything in cpu-mode.
                    // in_place = true;
                    custom_conv_ks1_inX_outX(1, m_synw[syn_wb_idx].data, m_synb[syn_wb_idx].data, m_target_img_h, m_target_img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, syn_out_i->origin(), !!frame_symbols.layers_synthesis[syn_idx].residual, !!frame_symbols.layers_synthesis[syn_idx].relu);
                }
#endif
            }
            if (fused)
            {
                // skip next step
                syn_idx += 1;
            }
            if (!in_place)
            {
                // swap our in & out.  accounting for any forced-first-time non-in-place.
                // we just update syn_in_i here, syn_out_i is calculated every loop.
                syn_in_i = syn_out_i;
            }
        }


        // track number of components.
        n_syn_in = n_syn_out;

        const auto time_this_syn_end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> this_syn_elapsed_seconds = time_this_syn_end - time_this_syn_start;
        if (m_verbosity >= 3)
            printf(" %g\n", (double)this_syn_elapsed_seconds.count());

        if (m_verbosity >= 100)
        {
            printf("SYNOUTPLANE0\n");
            syn_in_i->print_start(0, "SYNOUT0", 10, UPS_PRECISION);
        }
    }

    const auto time_syn_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> syn_elapsed_seconds = time_syn_end - time_syn_start;
    time_syn_seconds += syn_elapsed_seconds.count();

    // indicate correct number of final output planes.
    syn_in_i->update_to(syn_in_i->h, syn_in_i->w, syn_in_i->pad, n_syn_in);
    return syn_in_i;
}

frame_memory<SYN_INT_FLOAT> *cc_frame_decoder::run_syn(struct cc_bs_frame_coolchic &frame_symbols)
{
    // multiple synthesiser runs.
    // we have m_syn_1 containing upsampled output, ready to send to synthesisers.
    //
    // ASSUMING ALL SYNTHS POSSIBLE IN-PLACE.  IE, ALL KS ARE 1 or 3.
    // with n branches 1..n:
    // for branch1:
    //     synth m_syn_1 -> m_syn_2. (ie, 1st syn layer takes 1->2, remaining layers in 2)
    // for branch 2..n-1:
    //     synth m_syn_1 -> m_syn_3. (ie, 1st syn layer takes 1->3, remaining layers in 3)
    //     blend result m_syn_3 into m_syn_2.
    // for branch n:
    //     synth m_syn_1 in place. (ie, all layers in 1)
    //     blend result m_syn_1 into m_syn_2.
    //
    // result is m_syn_2.

    // IF NOT IN-PLACE (KS > 3)
    // with n branches 1..n:
    // for branch1:
    //     synth m_syn_1 -> m_syn_2 or m_syn_x. (ie, 1st syn layer 1->2, remaining layers in 2/x)
    // for branch 2..n-1:
    //     synth m_syn_1 -> m_syn_3 or m_syn_x. (ie, 1st syn layer 1->3, remaining layers in 3/x)
    //     blend result m_syn_3/x into m_syn_2.
    // for branch n:
    //     synth m_syn_1 in place or m_syn_x . (ie, all layers in 1)
    //     blend result m_syn_1 into m_syn_2.

    frame_memory<SYN_INT_FLOAT> *p_syn_1 = &m_syn_1; // initial upsample output -- preserved until last branch.
    frame_memory<SYN_INT_FLOAT> *p_syn_2 = &m_syn_2; // initial branch output, blended into by subsequent branches.
    frame_memory<SYN_INT_FLOAT> *p_syn_3 = &m_syn_3; // subsequence branch outputs
    frame_memory<SYN_INT_FLOAT> *p_syn_tmp = &m_syn_tmp;

    if (m_verbosity >= 100)
    {
        printf("SYNTHESIS INPUTS (%d)\n", frame_symbols.latent_n_2d_grid);
        for (int p = 0; p < frame_symbols.latent_n_2d_grid; p++)
        {
            printf("SYNPLANE %d ", p);
            m_syn_1.print_start(p, "SYNIN", -1, UPS_PRECISION);
        }
    }

    frame_memory<SYN_INT_FLOAT> *result = NULL;
    for (int b = 0; b < m_syn_n_branches; b++)
    {
        if (b == 0 && m_syn_n_branches > 1)
        {
            result = run_syn_branch(frame_symbols, b, p_syn_1, p_syn_2, p_syn_tmp);
            // result is either m_syn_2 or m_syn_tmp.
            if (result != p_syn_2)
            {
                // switch our notion of m_syn_2 and m_syn_tmp.
                p_syn_tmp = p_syn_2;
                p_syn_2 = result;
            }
        }
        else if (b < m_syn_n_branches-1)
        {
            result = run_syn_branch(frame_symbols, b, p_syn_1, p_syn_3, p_syn_tmp);
            // result is either m_syn_3 or m_syn_tmp.
            if (result != p_syn_3)
            {
                // switch our notion of m_syn_3 and m_syn_tmp.
                p_syn_tmp = p_syn_3;
                p_syn_3 = result;
            }
            if (b == 1)
                run_syn_blend2(frame_symbols, 3, b, p_syn_3, p_syn_2);
            else
                run_syn_blend1(frame_symbols, 3, b, p_syn_3, p_syn_2);
        }
        else
        {
            // NULL for out implies we can run in-place as much as we like, destroying p_syn_1.
            result = run_syn_branch(frame_symbols, b, p_syn_1, NULL, p_syn_tmp);
            // result is either m_syn_1 or m_syn_tmp.
            if (result != p_syn_1)
            {
                // switch our notion of m_syn_1 and m_syn_tmp.
                p_syn_tmp = p_syn_1;
                p_syn_1 = result;
            }
            if (b == 0)
            {
                ; // we don't have any blending, single branch.
            }
            else if (b == 1)
            {
                // 2 branches being blended to create result.
                run_syn_blend2(frame_symbols, 3, b, p_syn_1, p_syn_2);
                result = p_syn_2;
            }
            else
            {
                // subsequent branch being blended into result.
                run_syn_blend1(frame_symbols, 3, b, p_syn_1, p_syn_2);
                result = p_syn_2;
            }
        }
    }

    if (m_verbosity >= 100)
    {
        printf("SYNTHESIS OUTPUTS\n");
        for (int p = 0; p < result->planes; p++)
        {
            printf("SYNPLANE %d ", p);
            result->print_start(p, "SYNOUT", 10, UPS_PRECISION);
        }
    }

    return result;
}

// returned value should be transformed or otherwise copied before calling decode_frame again.
struct frame_memory<SYN_INT_FLOAT> *cc_frame_decoder::decode_frame(struct cc_bs_frame_coolchic &frame_symbols)
{
    read_arm(frame_symbols);
    read_ups(frame_symbols);
    read_syn(frame_symbols);

    if (m_verbosity >= 3)
    {
        printf("loaded weights\n");
        fflush(stdout);
    }

    check_allocations(frame_symbols);

    run_arm(frame_symbols);
    run_ups(frame_symbols);
    frame_memory<SYN_INT_FLOAT> *syn_result = run_syn(frame_symbols);

    return syn_result;
}

