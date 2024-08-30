
#include "common.h"
#include "frame-memory.h"

class cc_frame_decoder
{
public:
    cc_frame_decoder(struct cc_bs_gop_header &gop_header)
        : m_gop_header(gop_header),
          m_mlpw_t(NULL),
          m_mlpb(NULL),
          m_mlp_n_hidden_layers_arm(-1),
          m_synw(NULL),
          m_synb(NULL),
          m_syn_n_layers(-1)
          {};

    ~cc_frame_decoder() {delete[] m_mlpw_t; delete[] m_mlpb; delete[] m_synw; delete[] m_synb;};

public:
    struct cc_bs_gop_header &m_gop_header;

public:
    struct frame_memory *decode_frame(struct cc_bs_frame *frame_symbols);

private:
    weights_biases *m_mlpw_t;
    weights_biases *m_mlpb;
    weights_biases  m_mlpwOUT;
    weights_biases  m_mlpbOUT;
    int             m_mlp_n_hidden_layers_arm;

    weights_biases  m_upsw_t_4x4;

    weights_biases *m_synw;
    weights_biases *m_synb;
    int             m_syn_n_layers;

    buffer          m_syn3x3_linebuffer;

private:
    // for arm decode destination, and upsampling.
    std::vector<int> m_h_pyramid;
    std::vector<int> m_w_pyramid;
    std::vector<frame_memory> m_plane_pyramid;

private:
    int             m_arm_pad;
    int             m_ups_pad;
    int             m_max_pad;

    frame_memory    m_ups_1; // !!! really for_syn_0
    frame_memory    m_ups_2; // !!! really for_syn_1

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
    frame_memory   *run_syn(struct cc_bs_frame *frame_symbols);
};

