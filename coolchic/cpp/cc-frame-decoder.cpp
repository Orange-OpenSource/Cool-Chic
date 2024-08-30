
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
#include "syn_cpu.h"

extern float time_arm_seconds;
extern float time_ups_seconds;
extern float time_syn_seconds;

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

        result[i] = tmp;
    }
}

void decode_weights_qi(weights_biases &result, TDecBinCABAC *cabac, int count, int n_weights, int q_step_shift, int precision)
{
    result.update_to(n_weights);
    decode_weights_qi(result.data, cabac, count, n_weights, q_step_shift, precision);
}

// applying 4 small kernels, ks here is upsampling_kernel/2
//void custom_conv_ups_zxz1_cpu(int ks, int32_t *kw, int h_in, int w_in, int stride_in, int32_t *in, int32_t *out, int h_target, int w_target, int stride_out)
void custom_conv_ups_zxz1_cpu(int ks, int32_t *kw, int h_in, int w_in, int stride_in, int32_t *in, int stride_out, int32_t *out, int ups_mul_precision)
{
    int const kstride = 1;
    int offs0 = 0;

    for (int y = 0; y < h_in-ks+1; y += kstride, offs0 += stride_in)
    {
        for (int vf = 0; vf < 2; vf++, out += stride_out-(w_in-ks+1)*2) // vertical filter choice on this line.
        {
            int offs = offs0;
            for (int x = 0; x < w_in-ks+1; x += kstride, offs += kstride)
            {
                for (int hf = 0; hf < 2; hf++) // horizontal filter choice at this point
                {
                    int32_t sum = 0;
                    int32_t *k = kw+(vf*2+hf)*(ks*ks);
                    int offs2 = offs;
                    for (int yy = 0; yy < ks; yy++, offs2 += stride_in-ks)
                        for (int xx = 0; xx < ks; xx++)
                        {
                            sum += in[offs2++]*(*k++);
                        }
                    *out++ = sum >> ups_mul_precision;
                }
            }
        }
    }
}

// upsamplingx2: applies 4 smaller kernels sized [ks/2][ks/2]
// first_ups implies we are processing ARM output (.8) rather than upsampling output (.12).
void custom_upsample_4(int ks, int32_t *weightsx4xpose, int h_in, int w_in, int stride_in, int32_t *in, int stride_target, int32_t *out, bool first_ups)
{
#ifdef CCDECAPI_AVX2
    if (ks == 8)
    {
        if (first_ups)
            ups_4x4x4_fromarm_avx2(ks/2, weightsx4xpose, h_in, w_in, stride_in, in, stride_target, out);
        else
            ups_4x4x4_fromups_avx2(ks/2, weightsx4xpose, h_in, w_in, stride_in, in, stride_target, out);
    }
    else if (ks == 4)
    {
        if (first_ups)
            ups_4x2x2_fromarm_avx2(ks/2, weightsx4xpose, h_in, w_in, stride_in, in, stride_target, out);
        else
            ups_4x2x2_fromups_avx2(ks/2, weightsx4xpose, h_in, w_in, stride_in, in, stride_target, out);
    }
    else
#endif
    {
        custom_conv_ups_zxz1_cpu(ks/2, weightsx4xpose, h_in, w_in, stride_in, in, stride_target, out, first_ups ? ARM_PRECISION : UPS_PRECISION);
    }
}

void cc_frame_decoder::read_arm(struct cc_bs_frame *frame_symbols)
{
    struct cc_bs_frame_header &frame_header = frame_symbols->m_frame_header;
    if (frame_header.n_hidden_layers_arm != m_mlp_n_hidden_layers_arm)
    {
        m_mlp_n_hidden_layers_arm = frame_header.n_hidden_layers_arm;
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

    bs_fifo_weights = frame_symbols->m_arm_weights_hevc;
    cabac_weights.init(&bs_weights);
    cabac_weights.start();

    bs_fifo_biases = frame_symbols->m_arm_biases_hevc;
    cabac_biases.init(&bs_biases);
    cabac_biases.start();

    struct cc_bs_layer_quant_info &arm_lqi = frame_header.arm_lqi;

    int q_step_w_shift = Q_STEP_ARM_WEIGHT_SHIFT[arm_lqi.q_step_index_nn_weight];
    int q_step_b_shift = Q_STEP_ARM_BIAS_SHIFT[arm_lqi.q_step_index_nn_bias];
    int arm_n_features = frame_header.dim_arm;
    int32_t bucket[arm_n_features*arm_n_features]; // we transpose from this.

    for (int idx = 0; idx < frame_header.n_hidden_layers_arm; idx++)
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


void cc_frame_decoder::read_ups(struct cc_bs_frame *frame_symbols)
{
    struct cc_bs_frame_header &frame_header = frame_symbols->m_frame_header;

    int const ups_ks = frame_header.upsampling_kern_size;
    int32_t bucket[ups_ks*ups_ks]; // we transpose from this.

    if (frame_header.static_upsampling_kernel != 0)
    {
#if 0 // generate text.
        float bilinear_kernel[4 * 4] = {
            0.0625, 0.1875, 0.1875, 0.0625,
            0.1875, 0.5625, 0.5625, 0.1875,
            0.1875, 0.5625, 0.5625, 0.1875,
            0.0625, 0.1875, 0.1875, 0.0625
        };

        float bicubic_kernel[8 * 8] = {
             0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619,
             0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857,
            -0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498,
            -0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479,
            -0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479,
            -0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498,
             0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857,
             0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619
        };
        // generate text.
        printf("int32_t bilinear_kernel[4 * 4] = {\n");
        for (int ky = 0; ky < 4; ky++)
        {
            printf("    ");
            for (int kx = 0; kx < 4; kx++)
                printf("%d%s", (int32_t)(bilinear_kernel[ky*4+kx]*(1<<UPS_PRECISION)), ky == 3 && kx == 3 ? "" : ",");
            printf("\n");
        }
        printf("};\n");

        printf("int32_t bicubic_kernel[8 * 8] = {\n");
        for (int ky = 0; ky < 8; ky++)
        {
            printf("    ");
            for (int kx = 0; kx < 8; kx++)
                printf("%d%s", (int32_t)(bicubic_kernel[ky*8+kx]*(1<<UPS_PRECISION)), ky == 7 && kx == 7 ? "" : ",");
            printf("\n");
        }
        printf("};\n");
        exit(1);
#endif
        int32_t bilinear_kernel[4 * 4] = {
            256,768,768,256,
            768,2304,2304,768,
            768,2304,2304,768,
            256,768,768,256
        };
        int32_t bicubic_kernel[8 * 8] = {
            5,15,-37,-126,-126,-37,15,5,
            15,45,-113,-379,-379,-113,45,15,
            -37,-113,280,942,942,280,-113,-37,
            -126,-379,942,3164,3164,942,-379,-126,
            -126,-379,942,3164,3164,942,-379,-126,
            -37,-113,280,942,942,280,-113,-37,
            15,45,-113,-379,-379,-113,45,15,
            5,15,-37,-126,-126,-37,15,5
        };

        // Use bilinear static kernel if kernel size is smaller than 8, else bicubic
        int static_kernel_size = ups_ks < 8 ? 4 : 8;
        int32_t *static_kernel = ups_ks < 8 ? bilinear_kernel : bicubic_kernel;

        // First: re-init everything with zeros
        memset(&bucket[0], 0, ups_ks*ups_ks*sizeof(bucket[0]));

        // Then, write the values of the static filter
        // Top left coordinate of the first filter coefficient
        int top_left_coord = (ups_ks - static_kernel_size) / 2;
        for (int r = 0; r < static_kernel_size; r++)
        {
            for (int c = 0; c < static_kernel_size; c++)
            {
                bucket[(r + top_left_coord)*ups_ks + c  + top_left_coord] = static_kernel[r * static_kernel_size + c];
            }
        }
    }
    else
    {
        InputBitstream bs_weights;
        std::vector<UChar> &bs_fifo_weights = bs_weights.getFifo();

        TDecBinCABAC cabac_weights;

        bs_fifo_weights = frame_symbols->m_ups_weights_hevc;
        cabac_weights.init(&bs_weights);
        cabac_weights.start();
        struct cc_bs_layer_quant_info &ups_lqi = frame_header.ups_lqi;

        int q_step_w_shift = Q_STEP_UPS_SHIFT[ups_lqi.q_step_index_nn_weight];

        decode_weights_qi(&bucket[0], &cabac_weights, ups_lqi.scale_index_nn_weight, frame_header.upsampling_kern_size*frame_header.upsampling_kern_size, q_step_w_shift, UPS_PRECISION);
    }

    // transpose.
    int32_t bucket_t[ups_ks*ups_ks];

    int idx = 0;
    for (int y = 0; y < ups_ks; y++)
    {
        for (int x = 0; x < ups_ks; x++, idx++)
        {
            bucket_t[idx] = bucket[(ups_ks-1-y)*ups_ks+(ups_ks-1-x)];
        }
    }

    // extract 4 smaller filters. 8x8 -> 4x 4x4
    // f0 operates at origin, y=0, x=0
    // f1 operates at y=0, x=1
    // f2 operates at y=1, x=0
    // f3 operates at y=1, x=1
    m_upsw_t_4x4.update_to(ups_ks*ups_ks);
    int ups_ks2 = ups_ks/2;

    for (int f = 0; f < 4; f++)
    {
        int fbase_y = 1-f/2;
        int fbase_x = 1-f%2;
        for (int y = 0; y < ups_ks/2; y++)
        {
            for (int x = 0; x < ups_ks/2; x++)
            {
                m_upsw_t_4x4.data[f*ups_ks2*ups_ks2 + y*ups_ks2 + x] = bucket_t[(fbase_y+2*y)*ups_ks+fbase_x+2*x];
            }
        }
    }

}

void cc_frame_decoder::read_syn(struct cc_bs_frame *frame_symbols)
{
    struct cc_bs_frame_header &frame_header = frame_symbols->m_frame_header;

    if (frame_header.n_syn_layers != m_syn_n_layers)
    {
        m_syn_n_layers = frame_header.n_syn_layers;
        delete [] m_synw;
        delete [] m_synb;
        m_synw = new weights_biases[m_syn_n_layers];
        m_synb = new weights_biases[m_syn_n_layers];
    }

    InputBitstream bs_weights;
    std::vector<UChar> &bs_fifo_weights = bs_weights.getFifo();
    InputBitstream bs_biases;
    std::vector<UChar> &bs_fifo_biases = bs_biases.getFifo();

    TDecBinCABAC cabac_weights;
    TDecBinCABAC cabac_biases;

    bs_fifo_weights = frame_symbols->m_syn_weights_hevc;
    bs_fifo_biases = frame_symbols->m_syn_biases_hevc;
    cabac_weights.init(&bs_weights);
    cabac_weights.start();
    cabac_biases.init(&bs_biases);
    cabac_biases.start();
    struct cc_bs_layer_quant_info &syn_lqi = frame_header.syn_lqi;

    int q_step_w_shift = Q_STEP_SYN_WEIGHT_SHIFT[syn_lqi.q_step_index_nn_weight];
    int q_step_b_shift = Q_STEP_SYN_BIAS_SHIFT[syn_lqi.q_step_index_nn_bias];

    int n_in_ft = frame_header.n_latent_n_resolutions; // !!! features per layer
    for (int idx = 0; idx < frame_header.n_syn_layers; idx++)
    {
        struct cc_bs_syn_layer &syn = frame_header.layers_synthesis[idx];
        int n_weights = n_in_ft * syn.ks*syn.ks * syn.n_out_ft;
        int n_biases = syn.n_out_ft;

        decode_weights_qi(m_synw[idx], &cabac_weights, syn_lqi.scale_index_nn_weight, n_weights, q_step_w_shift, SYN_WEIGHT_PRECISION);
        decode_weights_qi(m_synb[idx], &cabac_biases,  syn_lqi.scale_index_nn_bias, n_biases,  q_step_b_shift, SYN_WEIGHT_PRECISION*2);

        n_in_ft = syn.n_out_ft;
    }
}

// check to see if the leading couple of syns are compatible with being fused
// together.
// 8, 16, 40 are checked.  8 is marginal (base mem allocation is 7, 8 is not much more).
bool cc_frame_decoder::can_fuse(struct cc_bs_frame *frame_symbols)
{
    auto &synthesis = frame_symbols->m_frame_header.layers_synthesis;

    bool fused = synthesis[0].ks == 1 && synthesis[1].ks == 1;
    if (!fused)
        return false;

    // Check for compatible numbers.
    int n_syn_in = frame_symbols->m_frame_header.n_latent_n_resolutions;
    int n_hidden = synthesis[0].n_out_ft;
    int n_syn_out = synthesis[1].n_out_ft;
    fused = (n_syn_in == 7)
                && (n_hidden == 8 || n_hidden == 16 || n_hidden == 40)
                && (n_syn_out == 3 || n_syn_out == 6 || n_syn_out == 9);
    return fused;
}

void cc_frame_decoder::check_allocations(struct cc_bs_frame *frame_symbols)
{
    struct cc_bs_frame_header &frame_header = frame_symbols->m_frame_header;

    // we allocate a max-sized (ignoring padding) buffer, suitable for holding
    // upsample final output and synthesis outputs during syn processing.
    int const latent_n_resolutions = frame_header.n_latent_n_resolutions;

    int n_max_planes = latent_n_resolutions;
    m_arm_pad = 4; // by how much the arm frame is padded.
    m_ups_pad = 8/2+1; // kernel size 8 upsampling.
    m_max_pad = std::max(m_arm_pad, m_ups_pad); // during arm, ups and syn. // !!! we use 5 for max ups (5) (ks=8)/2+1, and arm (4) (cxt_row_col)

    bool need_ups_2 = false; // we only need this if we cannot do in-place convolution.  Ie, ks 1 or 3 are in-place, otherwise not.
    for (int syn_idx = 0; syn_idx < frame_header.n_syn_layers; syn_idx++)
    {
        if (syn_idx == 0 && can_fuse(frame_symbols))
            continue; // ignore hidden layer size on fused layer.
        int n_syn_out = frame_header.layers_synthesis[syn_idx].n_out_ft;
        if (n_syn_out > n_max_planes)
            n_max_planes = n_syn_out;
        int n_pad = frame_header.layers_synthesis[syn_idx].ks/2;
        if (n_pad > m_max_pad)
            m_max_pad = n_pad;
        if (frame_header.layers_synthesis[syn_idx].ks != 1 && frame_header.layers_synthesis[syn_idx].ks != 3)
            need_ups_2 = true;
    }
    printf("MAX PLANES SET TO %d; MAX PAD SET TO %d\n", n_max_planes, m_max_pad);

    m_ups_1.update_to(m_gop_header.img_h, m_gop_header.img_w, n_max_planes, m_max_pad);
    // we do not (normally!) need m_ups_2.
    if (need_ups_2)
    {
        printf("ups_2 for syn allocated\n");
        m_ups_2.update_to(m_gop_header.img_h, m_gop_header.img_w, n_max_planes, m_max_pad);
    }

    // pyramid sizes.
    m_zero_layer.resize(frame_header.n_latent_n_resolutions);
    m_h_pyramid.resize(frame_header.n_latent_n_resolutions);
    m_w_pyramid.resize(frame_header.n_latent_n_resolutions);
    // pyramid storage -- enough pad for arm and ups.  no highest res layer.
    m_plane_pyramid.resize(frame_header.n_latent_n_resolutions);

    for (int layer_number = 0, h_grid = m_gop_header.img_h, w_grid = m_gop_header.img_w;
         layer_number < frame_header.n_latent_n_resolutions;
         layer_number++, h_grid = (h_grid+1)/2, w_grid = (w_grid+1)/2)
    {
        m_h_pyramid[layer_number] = h_grid;
        m_w_pyramid[layer_number] = w_grid;

        // we do not allocate the full-size.
        if (layer_number == 0)
            m_plane_pyramid[layer_number].update_to(0, 0, 0, 0); // not used.
        else
            m_plane_pyramid[layer_number].update_to(h_grid, w_grid, 1, m_max_pad);
    }
}

void cc_frame_decoder::run_arm(struct cc_bs_frame *frame_symbols)
{
    struct cc_bs_frame_header &frame_header = frame_symbols->m_frame_header;

    // RUN ARM
    // latent-layer processing.
    for (int layer_number = 0; layer_number < frame_header.n_latent_n_resolutions; layer_number++)
    {
        int const h_grid = m_h_pyramid[layer_number];
        int const w_grid = m_w_pyramid[layer_number];

        printf("starting layer %d\n", layer_number);
        fflush(stdout);

        auto &bytes = frame_symbols->m_latents_hevc[layer_number];
        if (bytes.size() == 0)
        {
            // produce a zero layer.
            m_zero_layer[layer_number] = true;
            continue;
        }
        m_zero_layer[layer_number] = false;

        frame_memory *dest = layer_number == 0 ? &m_ups_1 : &m_plane_pyramid[layer_number];
        dest->zero_pad(0, m_arm_pad);
        //dest->print_ranges("armin", 1, ARM_PRECISION);

        // BAC decoding:
        InputBitstream bsBAC;
        std::vector<UChar> &bs = bsBAC.getFifo();
        bs = bytes;

        TDecBinCABAC layerBAC;
        layerBAC.init(&bsBAC);
        layerBAC.start();

        BACContext bac_context;
        bac_context.set_layer(&layerBAC, h_grid, w_grid, frame_header.hls_sig_blksize);

        const auto time_arm_start = std::chrono::steady_clock::now();

#ifdef CCDECAPI_AVX2
        if (frame_header.dim_arm == 8)
            custom_conv_11_int32_avx2_8_X_X(
                                      m_mlpw_t, m_mlpb,
                                      &m_mlpwOUT, &m_mlpbOUT,
                                      get_context_indicies(frame_header.dim_arm, dest->stride), frame_header.dim_arm, frame_header.n_hidden_layers_arm,
                                      dest->origin(),
                                      h_grid, w_grid, (dest->stride-w_grid)/2,
                                      bac_context
                                      );
        else if (frame_header.dim_arm == 16)
            custom_conv_11_int32_avx2_16_X_X(
                                      m_mlpw_t, m_mlpb,
                                      &m_mlpwOUT, &m_mlpbOUT,
                                      get_context_indicies(frame_header.dim_arm, dest->stride), frame_header.dim_arm, frame_header.n_hidden_layers_arm,
                                      dest->origin(),
                                      h_grid, w_grid, (dest->stride-w_grid)/2,
                                      bac_context
                                      );
        else if (frame_header.dim_arm == 24)
            custom_conv_11_int32_avx2_24_X_X(
                                      m_mlpw_t, m_mlpb,
                                      &m_mlpwOUT, &m_mlpbOUT,
                                      get_context_indicies(frame_header.dim_arm, dest->stride), frame_header.dim_arm, frame_header.n_hidden_layers_arm,
                                      dest->origin(),
                                      h_grid, w_grid, (dest->stride-w_grid)/2,
                                      bac_context
                                      );
        else if (frame_header.dim_arm == 32)
            custom_conv_11_int32_avx2_32_X_X(
                                      m_mlpw_t, m_mlpb,
                                      &m_mlpwOUT, &m_mlpbOUT,
                                      get_context_indicies(frame_header.dim_arm, dest->stride), frame_header.dim_arm, frame_header.n_hidden_layers_arm,
                                      dest->origin(),
                                      h_grid, w_grid, (dest->stride-w_grid)/2,
                                      bac_context
                                      );
        else
#endif
            custom_conv_11_int32_cpu_X_X_X(
                                      m_mlpw_t, m_mlpb,
                                      &m_mlpwOUT, &m_mlpbOUT,
                                      get_context_indicies(frame_header.dim_arm, dest->stride), frame_header.dim_arm, frame_header.n_hidden_layers_arm,

                                      dest->origin(),
                                      h_grid, w_grid, (dest->stride-w_grid)/2,

                                      bac_context
                                      );

        //dest->print_ranges("armout", 1, ARM_PRECISION);

#if UPS_PRECISION > ARM_PRECISION
        if (layer_number == 0)
        {
            // in-place precision change for high-resolution layer.
            // upsampling is not used here, which normally changes the .8 to .12.
            // we do it by hand.
            int32_t *src = dest->plane_origin(0);
            int const lshift = UPS_PRECISION-ARM_PRECISION;
            for (int y = 0; y < h_grid; y++, src += dest->stride-w_grid)
            {
                for (int x = 0; x < w_grid; x++)
                {
                    *src++ <<= lshift;
                }
            }
#else
            what system is this
#endif
        }

        const auto time_arm_done = std::chrono::steady_clock::now();
        const std::chrono::duration<double> arm_elapsed = (time_arm_done-time_arm_start);
        time_arm_seconds += (float)arm_elapsed.count();

    } // layer.

    printf("arm done!\n");
}

void cc_frame_decoder::run_ups(struct cc_bs_frame *frame_symbols)
{
    struct cc_bs_frame_header &frame_header = frame_symbols->m_frame_header;

    // NEW UPSAMPLE
    const auto time_ups_start = std::chrono::steady_clock::now();

    int ups_ks = frame_header.upsampling_kern_size;

    for (int layer_number = 0, h_grid = m_gop_header.img_h, w_grid = m_gop_header.img_w;
         layer_number < frame_header.n_latent_n_resolutions;
         layer_number++, h_grid = (h_grid+1)/2, w_grid = (w_grid+1)/2)
    {
        if (m_zero_layer[layer_number])
        {
            // no need to upsample.  just zero the final content.
            m_ups_1.zero_plane_content(layer_number);
            continue;
        }
        if (layer_number == 0)
        {
            // full res. if non-null, already present in m_ups_layers[0].
            continue;
        }

        // continually scale up to the final resolution.  We upsample through lower layer numbers,
        // which have already been done until the final full res destination in m_ups_1.
        for (int target_layer = layer_number-1; target_layer >= 0; target_layer--)
        {
            frame_memory *dest;
            int dest_plane;
            if (target_layer == 0)
            {
                dest = &m_ups_1;
                dest_plane = layer_number;
            }
            else
            {
                dest = &m_plane_pyramid[target_layer];
                dest_plane = 0;
            }
            frame_memory *lo_res = &m_plane_pyramid[target_layer+1];
            int pad = ups_ks/2/2;
            lo_res->custom_pad_replicate_plane_in_place_i(0, pad);
            //printf("upslayer%dtarget%d ", layer_number, target_layer); lo_res->print_ranges("ups_in", 1, target_layer == layer_number-1 ? ARM_PRECISION : UPS_PRECISION);
            // our output is not the direct origin -- we need an extra crop to point at pad start.
            custom_upsample_4(ups_ks, m_upsw_t_4x4.data, m_h_pyramid[target_layer+1]+2*pad, m_w_pyramid[target_layer+1]+2*pad, lo_res->stride,
                              lo_res->pad_origin(pad), dest->stride, dest->pad_origin(dest_plane, 1), target_layer == layer_number-1);
        }
    }

    const auto time_ups_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> hand_ups_elapsed_seconds = time_ups_end - time_ups_start;
    time_ups_seconds += hand_ups_elapsed_seconds.count();
}

// result points to either m_ups_1 or m_ups_2
frame_memory *cc_frame_decoder::run_syn(struct cc_bs_frame *frame_symbols)
{
    struct cc_bs_frame_header &frame_header = frame_symbols->m_frame_header;
    // NEW SYNTHESIS: IN-PLACE CONV

    int    n_syn_in = frame_header.n_latent_n_resolutions;
    frame_memory *syn_in_i = &m_ups_1;
    frame_memory *syn_out_i = &m_ups_2; // if cannot do syn in-place.
    const auto time_syn_start = std::chrono::steady_clock::now();

    for (int syn_idx = 0; syn_idx < frame_header.n_syn_layers; syn_idx++)
    {
        int n_syn_out = frame_header.layers_synthesis[syn_idx].n_out_ft;
        printf("SYN%d: k=%d #in=%d #out=%d", syn_idx, frame_header.layers_synthesis[syn_idx].ks, n_syn_in, n_syn_out);
        auto time_this_syn_start = std::chrono::steady_clock::now();

        //syn_in_i->print_ranges("syn_in", n_syn_in, SYN_MUL_PRECISION);
        if (frame_header.layers_synthesis[syn_idx].ks > 1)
        {
            // pad syn_in, and direct output to syn_out.
            int pad = frame_header.layers_synthesis[syn_idx].ks/2;
            int w_padded = m_gop_header.img_w+2*pad;
            int h_padded = m_gop_header.img_h+2*pad;
            int origin_idx = syn_in_i->origin_idx;
            int pad_origin_idx = origin_idx-pad*syn_in_i->stride-pad;
            for (int l = 0; l < n_syn_in; l++)
                syn_in_i->custom_pad_replicate_plane_in_place_i(l, pad);
            printf(" padded ");

#ifdef CCDECAPI_AVX2
            bool in_place = false;
            if (frame_header.layers_synthesis[syn_idx].ks == 3)
            {
                // optimized 3x3 with line buffer.
                in_place = true;
                m_syn3x3_linebuffer.update_to(2*n_syn_out*m_gop_header.img_w); // two lines for now, could eventually be 1 line plus a few pixels.
                if (n_syn_in == 3 && n_syn_out == 3)
                    custom_conv_ks3_in3_out3_lb_avx2(frame_header.layers_synthesis[syn_idx].ks, m_synw[syn_idx].data, m_synb[syn_idx].data,
                                                    h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, NULL,
                                                    m_syn3x3_linebuffer.data,
                                                    !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
                else if (n_syn_in == 6 && n_syn_out == 6)
                    custom_conv_ks3_in6_out6_lb_avx2(frame_header.layers_synthesis[syn_idx].ks, m_synw[syn_idx].data, m_synb[syn_idx].data,
                                                    h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, NULL,
                                                    m_syn3x3_linebuffer.data,
                                                    !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
                else if (n_syn_in == 9 && n_syn_out == 6)
                    custom_conv_ks3_in9_out6_lb_avx2(frame_header.layers_synthesis[syn_idx].ks, m_synw[syn_idx].data, m_synb[syn_idx].data,
                                                    h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, NULL,
                                                    m_syn3x3_linebuffer.data,
                                                    !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
                else if (n_syn_in == 9 && n_syn_out == 9)
                    custom_conv_ks3_in9_out9_lb_avx2(frame_header.layers_synthesis[syn_idx].ks, m_synw[syn_idx].data, m_synb[syn_idx].data,
                                                    h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, NULL,
                                                    m_syn3x3_linebuffer.data,
                                                    !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
                else
                    custom_conv_ks3_inX_outX_lb_avx2(frame_header.layers_synthesis[syn_idx].ks, m_synw[syn_idx].data, m_synb[syn_idx].data,
                                                    h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, NULL,
                                                    m_syn3x3_linebuffer.data,
                                                    !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            }
            else
                custom_conv_ksX_inX_outX_avx2(frame_header.layers_synthesis[syn_idx].ks, m_synw[syn_idx].data, m_synb[syn_idx].data,
                                                h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, syn_out_i->origin(),
                                                !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
#else
            bool in_place = false;
            if (frame_header.layers_synthesis[syn_idx].ks == 3)
            {
                in_place = true;
                m_syn3x3_linebuffer.update_to(2*n_syn_out*m_gop_header.img_w); // two lines for now, could eventually be 1 line plus a few pixels.
                custom_conv_ks3_inX_outX_lb(frame_header.layers_synthesis[syn_idx].ks, m_synw[syn_idx].data, m_synb[syn_idx].data,
                                                h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, NULL,
                                                m_syn3x3_linebuffer.data,
                                                !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            }
            else
                custom_conv_ksX_inX_outX(frame_header.layers_synthesis[syn_idx].ks, m_synw[syn_idx].data, m_synb[syn_idx].data,
                                                h_padded, w_padded, syn_in_i->stride, syn_in_i->plane_stride, origin_idx-pad_origin_idx, n_syn_in, syn_in_i->raw()+pad_origin_idx, n_syn_out, syn_out_i->origin(),
                                                !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
#endif

            if (!in_place)
            {
            // swap our in & out.
            frame_memory *tmp = syn_in_i;
            syn_in_i = syn_out_i;
            syn_out_i = tmp;
            }
        }
        else
        {
            // possible in-place if ATATIME is big enough to buffer all outputs for single spatial pixel.
            // assuming max 4 here (avx2) and infinite (cpu)
            bool in_place =  n_syn_out <= 4; // !!! hardwired, not too many outputs.
            // possible fusion.
            bool fused = syn_idx == 0 && can_fuse(frame_symbols);
            if (fused)
            {
                // compatible!
                in_place = true;
                int n_hidden = n_syn_out;
                n_syn_out = frame_header.layers_synthesis[syn_idx+1].n_out_ft;

                // transpose.
                int32_t synw_fused[n_syn_in*n_hidden];
                int kidx = 0;
                for (int kx = 0; kx < n_syn_in; kx++)
                    for (int ky = 0; ky < n_hidden; ky++)
                    {
                        synw_fused[kidx++] = m_synw[syn_idx].data[ky*n_syn_in+kx];
                    }
#ifdef CCDECAPI_AVX2
                if (n_hidden == 40)
                {
                    if (n_syn_out == 3)
                        custom_conv_ks1_in7_hidden40_out3_avx2(1, synw_fused, m_synb[syn_idx].data, m_synw[syn_idx+1].data, m_synb[syn_idx+1].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_in_i->origin());
                    else if (n_syn_out == 6)
                        custom_conv_ks1_in7_hidden40_out6_avx2(1, synw_fused, m_synb[syn_idx].data, m_synw[syn_idx+1].data, m_synb[syn_idx+1].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_in_i->origin());
                    else
                        custom_conv_ks1_in7_hidden40_out9_avx2(1, synw_fused, m_synb[syn_idx].data, m_synw[syn_idx+1].data, m_synb[syn_idx+1].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_in_i->origin());
                }
                else if (n_hidden == 16)
                    custom_conv_ks1_in7_hidden16_out3_avx2(1, synw_fused, m_synb[syn_idx].data, m_synw[syn_idx+1].data, m_synb[syn_idx+1].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_in_i->origin());
                else
                    custom_conv_ks1_in7_hidden8_out3_avx2(1, synw_fused, m_synb[syn_idx].data, m_synw[syn_idx+1].data, m_synb[syn_idx+1].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_in_i->origin());
#else
                custom_conv_ks1_inX_hiddenX_outX(1, synw_fused, m_synb[syn_idx].data, m_synw[syn_idx+1].data, m_synb[syn_idx+1].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, n_syn_in, n_hidden, syn_in_i->origin(), n_syn_out, syn_in_i->origin());
#endif
            }
            else
            {
                // 1x1 kernel but not fused.
#ifdef CCDECAPI_AVX2
            if (n_syn_in == 7 && n_syn_out == 9)
                custom_conv_ks1_in7_out9_avx2(1, m_synw[syn_idx].data, m_synb[syn_idx].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, in_place ? syn_in_i->origin() : syn_out_i->origin(), !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else if (n_syn_in == 9 && n_syn_out == 3)
                custom_conv_ks1_in9_out3_avx2(1, m_synw[syn_idx].data, m_synb[syn_idx].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, in_place ? syn_in_i->origin() : syn_out_i->origin(), !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else if (n_syn_in == 9 && n_syn_out == 6)
                custom_conv_ks1_in9_out6_avx2(1, m_synw[syn_idx].data, m_synb[syn_idx].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, in_place ? syn_in_i->origin() : syn_out_i->origin(), !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else if (n_syn_in == 9 && n_syn_out == 9)
                custom_conv_ks1_in9_out9_avx2(1, m_synw[syn_idx].data, m_synb[syn_idx].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, in_place ? syn_in_i->origin() : syn_out_i->origin(), !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else if (n_syn_in == 7 && n_syn_out == 40)
                custom_conv_ks1_in7_out40_avx2(1, m_synw[syn_idx].data, m_synb[syn_idx].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, in_place ? syn_in_i->origin() : syn_out_i->origin(), !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else if (n_syn_in == 7 && n_syn_out == 16)
                custom_conv_ks1_in7_out16_avx2(1, m_synw[syn_idx].data, m_synb[syn_idx].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, in_place ? syn_in_i->origin() : syn_out_i->origin(), !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else if (                 n_syn_out == 3)
                custom_conv_ks1_inX_out3_avx2(1, m_synw[syn_idx].data, m_synb[syn_idx].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, in_place ? syn_in_i->origin() : syn_out_i->origin(), !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else if (                 n_syn_out == 6)
                custom_conv_ks1_inX_out6_avx2(1, m_synw[syn_idx].data, m_synb[syn_idx].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, in_place ? syn_in_i->origin() : syn_out_i->origin(), !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else if (                 n_syn_out == 9)
                custom_conv_ks1_inX_out9_avx2(1, m_synw[syn_idx].data, m_synb[syn_idx].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, in_place ? syn_in_i->origin() : syn_out_i->origin(), !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
            else
                custom_conv_ks1_inX_outX_avx2(1, m_synw[syn_idx].data, m_synb[syn_idx].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, in_place ? syn_in_i->origin() : syn_out_i->origin(), !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
#else
            in_place = true;
            custom_conv_ks1_inX_outX(1, m_synw[syn_idx].data, m_synb[syn_idx].data, m_gop_header.img_h, m_gop_header.img_w, syn_in_i->stride, syn_in_i->plane_stride, 0, n_syn_in, syn_in_i->origin(), n_syn_out, in_place ? syn_in_i->origin() : syn_out_i->origin(), !!frame_header.layers_synthesis[syn_idx].residual, !!frame_header.layers_synthesis[syn_idx].relu);
#endif
            }
            if (fused)
            {
                // skip next step
                syn_idx += 1;
            }
            if (!in_place)
            {
                // swap our in & out.
                frame_memory *tmp = syn_in_i;
                syn_in_i = syn_out_i;
                syn_out_i = tmp;
            }
        }


        // track number of components.
        n_syn_in = n_syn_out;

        const auto time_this_syn_end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> this_syn_elapsed_seconds = time_this_syn_end - time_this_syn_start;
        printf(" %g\n", (double)this_syn_elapsed_seconds.count());
    }
    //syn_in_i->print_ranges("final", n_syn_in, SYN_MUL_PRECISION);

    const auto time_syn_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> syn_elapsed_seconds = time_syn_end - time_syn_start;
    time_syn_seconds += syn_elapsed_seconds.count();

    return syn_in_i;
}

// returned value should be transformed or otherwise copied before calling decode_frame again.
struct frame_memory *cc_frame_decoder::decode_frame(struct cc_bs_frame *frame_symbols)
{
    read_arm(frame_symbols);
    read_ups(frame_symbols);
    read_syn(frame_symbols);

    printf("loaded weights\n");
    fflush(stdout);

    check_allocations(frame_symbols);

    run_arm(frame_symbols);
    run_ups(frame_symbols);
    frame_memory *syn_result = run_syn(frame_symbols);

    return syn_result;
}

