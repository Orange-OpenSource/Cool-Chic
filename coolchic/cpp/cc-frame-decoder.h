
#include "common.h"
#include "frame-memory.h"
#include "common_randomness.h"

class cc_frame_decoder
{
public:
    cc_frame_decoder(struct cc_bs_gop_header &gop_header, int output_bitdepth, int output_chroma_format
#if defined(CCDECAPI_AVX2_OPTIONAL)
                     , bool use_avx2
#endif
                     , int verbosity
                     )
        : m_gop_header(gop_header),
          m_output_bitdepth(output_bitdepth),
          m_output_chroma_format(output_chroma_format),
          m_frame_data_type(gop_header.frame_data_type),
          m_mlpw_t(NULL),
          m_mlpb(NULL),
          m_mlp_n_hidden_layers_arm(-1),
          m_ups_n(-1),
          m_upsw(NULL),
          m_ups_n_preconcat(-1),
          m_upsw_preconcat(NULL),
          m_syn_n_branches(1), // historic -- only single branch now.
          m_syn_n_layers(-1),
          m_synw(NULL),
          m_synb(NULL)
#if defined(CCDECAPI_AVX2)
#if defined(CCDECAPI_AVX2_OPTIONAL)
          ,m_use_avx2(use_avx2)
#else
          ,m_use_avx2(true)
#endif
#endif
          ,m_verbosity(verbosity)
          ,m_arm_pad(0)
          ,m_ups_pad(0)
          ,m_max_pad(0)
          ,m_target_img_h(0)
          ,m_target_img_w(0)
          { 
            if (m_output_bitdepth == 0)
                m_output_bitdepth = m_gop_header.output_bitdepth;
            if (m_output_chroma_format == 0)
                m_output_chroma_format = m_gop_header.frame_data_type == 1 ? 420 : 444;
          };

    ~cc_frame_decoder() {delete[] m_mlpw_t; delete[] m_mlpb; delete[] m_upsw; delete[] m_upsw_preconcat; delete[] m_synw; delete[] m_synb;};
public:
    int get_syn_idx(int branch_idx, int layer_idx) { return branch_idx*m_syn_n_layers + layer_idx; }

public:
    struct cc_bs_gop_header &m_gop_header;
    int m_output_bitdepth;
    int m_output_chroma_format;
    int m_frame_data_type;

public:
    struct frame_memory<SYN_INT_FLOAT> *decode_frame(struct cc_bs_frame_coolchic &frame_symbols);

private:
    weights_biases *m_mlpw_t;
    weights_biases *m_mlpb;
    weights_biases  m_mlpwOUT;
    weights_biases  m_mlpbOUT;
    int             m_mlp_n_hidden_layers_arm;

    int             m_ups_n;
    weights_biases_ups *m_upsw;
    int             m_ups_n_preconcat;
    weights_biases_ups *m_upsw_preconcat;

    int             m_syn_n_branches;
    int             m_syn_n_layers;
    weights_biases  m_syn_blends; // m_syn_n_branches blend values.
    weights_biases_syn *m_synw; // [branchidx*m_syn_n_layers+synidx]
    weights_biases_syn *m_synb; // [branchidx*m_syn_n_layers+synidx]

    buffer<SYN_INT_FLOAT> m_syn3x3_linebuffer;

#if defined(CCDECAPI_AVX2)
    bool            m_use_avx2;
#endif
    int             m_verbosity;

    common_randomness   cr;
private:
    // for arm decode destination, and upsampling.
    std::vector<int> m_h_pyramid;
    std::vector<int> m_w_pyramid;
    std::vector<frame_memory<int32_t>> m_plane_pyramid;

    // frame_memory<UPS_INT_FLOAT> m_noise_single_source; // upsample from here, layer by layer.
    //frame_memory<UPS_INT_FLOAT> m_noise_upsampled_block; // total upsampled result. !!! now m_syn_1 for image.

    // !!! temporary -- when ups is float.
    std::vector<frame_memory<UPS_INT_FLOAT>> m_plane_pyramid_for_ups;

private:
    int             m_arm_pad;
    int             m_ups_pad;
    int             m_max_pad;

    // we can now send a downsampled image through syn.
    int             m_target_img_h;
    int             m_target_img_w;

    frame_memory<UPS_INT_FLOAT>    m_ups_h2w2; // internal refinement target prior to upsample.
    frame_memory<UPS_INT_FLOAT>    m_ups_hw;  // during 2-pass refinement and 2-padd upsample.
    frame_memory<UPS_INT_FLOAT>  m_pre_syn_1; // !!! temporary -- int upsample output, to convert to float m_syn_1.
    frame_memory<SYN_INT_FLOAT>    m_syn_1;
    frame_memory<SYN_INT_FLOAT>    m_syn_2;
    frame_memory<SYN_INT_FLOAT>    m_syn_3;
    frame_memory<SYN_INT_FLOAT>    m_syn_tmp;

private:
    // set by arm to indicate no content, avoid ups for no content.
    std::vector<bool> m_zero_layer;

private:
    //{{NOISE
    void            ups_bicubic(frame_memory<UPS_INT_FLOAT> &frame_src, frame_memory<UPS_INT_FLOAT> &frame_dst, int dst_plane, int h_dst, int w_dst, bool dst_ready, frame_memory<UPS_INT_FLOAT> &frame_tmp);
    void            noise_fill_and_upsample_from_layer(int start_layer_number, int &noise_val_idx_base, int dst_plane);
    //NOISE}}

private:
    void            read_arm(struct cc_bs_frame_coolchic &frame_symbols);
    void            read_ups(struct cc_bs_frame_coolchic &frame_symbols);
    void            read_syn(struct cc_bs_frame_coolchic &frame_symbols);
    bool            can_fuse(struct cc_bs_frame_coolchic &frame_symbols);
    void            check_allocations(struct cc_bs_frame_coolchic &frame_symbols);

    void            run_arm(struct cc_bs_frame_coolchic &frame_symbols);
    void            run_ups(struct cc_bs_frame_coolchic &frame_symbols);
    frame_memory<SYN_INT_FLOAT>   *run_syn_blend1(struct cc_bs_frame_coolchic &frame_symbols, int n_planes, int branch_no, frame_memory<SYN_INT_FLOAT> *syn_in, frame_memory<SYN_INT_FLOAT> *syn_out);
    frame_memory<SYN_INT_FLOAT>   *run_syn_blend2(struct cc_bs_frame_coolchic &frame_symbols, int n_planes, int branch_no, frame_memory<SYN_INT_FLOAT> *syn_in, frame_memory<SYN_INT_FLOAT> *syn_out);
    frame_memory<SYN_INT_FLOAT>   *run_syn_branch(struct cc_bs_frame_coolchic &frame_symbols, int branch_no, frame_memory<SYN_INT_FLOAT> *syn_in, frame_memory<SYN_INT_FLOAT> *syn_out, frame_memory<SYN_INT_FLOAT> *syn_tmp);
    frame_memory<SYN_INT_FLOAT>   *run_syn(struct cc_bs_frame_coolchic &frame_symbols);
};

