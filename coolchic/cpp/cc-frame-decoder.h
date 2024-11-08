
#include "common.h"
#include "frame-memory.h"

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
          m_mlpw_t(NULL),
          m_mlpb(NULL),
          m_mlp_n_hidden_layers_arm(-1),
          m_ups_n(-1),
          m_upsw(NULL),
          m_ups_n_preconcat(-1),
          m_upsw_preconcat(NULL),
          m_syn_n_branches(-1),
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
          { 
            if (m_output_bitdepth == 0)
                m_output_bitdepth = m_gop_header.bitdepth;
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

public:
    struct frame_memory *decode_frame(struct cc_bs_frame *frame_symbols);

private:
    weights_biases *m_mlpw_t;
    weights_biases *m_mlpb;
    weights_biases  m_mlpwOUT;
    weights_biases  m_mlpbOUT;
    int             m_mlp_n_hidden_layers_arm;

    int             m_ups_n;
    weights_biases *m_upsw;
    int             m_ups_n_preconcat;
    weights_biases *m_upsw_preconcat;

    int             m_syn_n_branches;
    int             m_syn_n_layers;
    weights_biases  m_syn_blends; // m_syn_n_branches blend values.
    weights_biases *m_synw; // [branchidx*m_syn_n_layers+synidx]
    weights_biases *m_synb; // [branchidx*m_syn_n_layers+synidx]

    buffer          m_syn3x3_linebuffer;

#if defined(CCDECAPI_AVX2)
    bool            m_use_avx2;
#endif
    int             m_verbosity;

private:
    // for arm decode destination, and upsampling.
    std::vector<int> m_h_pyramid;
    std::vector<int> m_w_pyramid;
    std::vector<frame_memory> m_plane_pyramid;

private:
    int             m_arm_pad;
    int             m_ups_pad;
    int             m_max_pad;

    frame_memory    m_ups_h2w2; // internal refinement target prior to upsample.
    frame_memory    m_ups_hw;  // during 2-pass refinement and 2-padd upsample.
    frame_memory    m_syn_1;
    frame_memory    m_syn_2;
    frame_memory    m_syn_3;
    frame_memory    m_syn_tmp;

private:
    // set by arm to indicate no content, avoid ups for no content.
    std::vector<bool> m_zero_layer;

private:
    void            read_arm(struct cc_bs_frame *frame_symbols);
    void            read_ups(struct cc_bs_frame *frame_symbols);
    void            read_syn(struct cc_bs_frame *frame_symbols);
    bool            can_fuse(struct cc_bs_frame *frame_symbols);
    void            check_allocations(struct cc_bs_frame *frame_symbols);

    void            run_arm(struct cc_bs_frame *frame_symbols);
    void            run_ups(struct cc_bs_frame *frame_symbols);
    frame_memory   *run_syn_blend1(struct cc_bs_frame *frame_symbols, int n_planes, int branch_no, frame_memory *syn_in, frame_memory *syn_out);
    frame_memory   *run_syn_blend2(struct cc_bs_frame *frame_symbols, int n_planes, int branch_no, frame_memory *syn_in, frame_memory *syn_out);
    frame_memory   *run_syn_branch(struct cc_bs_frame *frame_symbols, int branch_no, frame_memory *syn_in, frame_memory *syn_out, frame_memory *syn_tmp);
    frame_memory   *run_syn(struct cc_bs_frame *frame_symbols);
};

