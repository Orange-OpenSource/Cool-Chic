
/*
 * USED WITH
 * #define CCDECAPI_CPU
 * or
 * #define CCDECAPI_AVX2
 * or
 * #define CCDECAPI_AVX2_OPTIONAL
 */

#if defined(CCDECAPI_AVX2_OPTIONAL)
#define CCDECAPI_AVX2
#endif

#include <stdlib.h>
#include <iostream>
#include <memory>
#include <chrono> // timing.
#include <math.h>
#include <sstream>

// layer 1:  operate at layer: 6.10, weights 4.12, mul is (A*B) -> 10.22: HI is 10.6; sum 10.6, out 10.6.
// layer >1: operate at layer: 4.12, weights 4.12, mul is (A*B) -> 8.24: HI is 8.8, sum is 8.8, out 8.8.

#include "TDecBinCoderCABAC.h"

#include "common.h"
#include "cc-contexts.h" // Emitted from python extraction, same for all extracts.
#include "cc-bitstream.h"

#include "cc-frame-decoder.h"
#include "warp.h"

float time_bac_seconds = 0.0;
float time_arm_seconds = 0.0;
float time_ups_seconds = 0.0;
float time_syn_seconds = 0.0;
float time_blend_seconds = 0.0;
float time_warp_seconds = 0.0;
float time_warp_coeffgen_seconds = 0.0;
float time_warp_coeffget_seconds = 0.0;
float time_bpred_seconds = 0.0;
float time_all_seconds = 0.0;

// like std::byteswap, but that's only in c++23 onwards!
inline unsigned short byteswap(unsigned short x)
{
    unsigned short hi = x&0xFF00;
    unsigned short lo = x&0x00FF;
    return (lo<<8)|(hi>>8);
}

// like ends_with in c++20 and onwards.
inline bool ends_with(const std::string& a, const std::string& b)
{
    if (b.size() > a.size())
        return false;
    return std::equal(a.begin() + a.size() - b.size(), a.end(), b.begin());
}

// onoly use nc == 3
template <typename P>
void ppm_out(int nc, int bit_depth, struct frame_memory<P> &in_info, FILE *fout)
{
    int const h = in_info.h;
    int const w = in_info.w;
    int const stride = in_info.stride;
    int const plane_stride = in_info.plane_stride;
    int const max_sample_val = (1<<bit_depth)-1;
    P *ins = in_info.origin();
    //printf("writing %d-bit ppm\n", bit_depth);

    if (nc != 3)
    {
        printf("only 3-color image output supported for the moment\n");
        exit(1);
    }

    fprintf(fout, "P6\n");
    fprintf(fout, "%d %d\n", w, h);
    fprintf(fout, "%d\n", max_sample_val);
    P *inR = ins;
    P *inG = inR+plane_stride;
    P *inB = inG+plane_stride;

    if (bit_depth <= 8)
    {
        for (int y = 0; y < h; y++, inR += stride-w, inG += stride-w, inB += stride-w)
            for (int x = 0; x < w; x++)
            {
                int r;
                int g;
                int b;

                if constexpr(std::is_same<P, int32_t>::value)
                {
                    // precision is SYN_LAYER_PRECISION
                    r = ((*inR++)*max_sample_val+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
                    g = ((*inG++)*max_sample_val+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
                    b = ((*inB++)*max_sample_val+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
                }
                else
                {
                    r = (int)((*inR++)*max_sample_val + 0.5);
                    g = (int)((*inG++)*max_sample_val + 0.5);
                    b = (int)((*inB++)*max_sample_val + 0.5);
                }
                if (r < 0) r = 0;
                if (g < 0) g = 0;
                if (b < 0) b = 0;
                if (r > max_sample_val) r = max_sample_val;
                if (g > max_sample_val) g = max_sample_val;
                if (b > max_sample_val) b = max_sample_val;
                unsigned char pix[3];
                pix[0] = r;
                pix[1] = g;
                pix[2] = b;
                fwrite(pix, 3, sizeof(pix[0]), fout);
            }
    }
    else
    {
        for (int y = 0; y < h; y++, inR += stride-w, inG += stride-w, inB += stride-w)
            for (int x = 0; x < w; x++)
            {
                int r;
                int g;
                int b;

                if constexpr(std::is_same<P, int32_t>::value)
                {
                    // precision is SYN_LAYER_PRECISION
                    r = ((*inR++)*max_sample_val+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
                    g = ((*inG++)*max_sample_val+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
                    b = ((*inB++)*max_sample_val+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
                }
                else
                {
                    r = (int)((*inR++)*max_sample_val + 0.5);
                    g = (int)((*inG++)*max_sample_val + 0.5);
                    b = (int)((*inB++)*max_sample_val + 0.5);
                }
                if (r < 0) r = 0;
                if (g < 0) g = 0;
                if (b < 0) b = 0;
                if (r > max_sample_val) r = max_sample_val;
                if (g > max_sample_val) g = max_sample_val;
                if (b > max_sample_val) b = max_sample_val;
                unsigned short pix[3];
                pix[0] = byteswap(r);
                pix[1] = byteswap(g);
                pix[2] = byteswap(b);
                fwrite(pix, 3, sizeof(pix[0]), fout);
            }
    }

    fflush(fout);
}

// we 'stabilise' the yuv to (0..255) as integers, as well as uv subsample.
// incoming 444 is planar, outgoing 420 is planar.
// A non-NULL pointer (*out) is assumed to have the good size!
template <typename P, typename POUT, int max_val>
void convert_444_420_xb(POUT **param_out, struct frame_memory<P> const &in)
{
    int const h = in.h;
    int const w = in.w;
    int const in_stride = in.stride;
    if (*param_out == NULL)
        *param_out = new POUT[h*w*3/2];
    POUT *out = *param_out;
    P const *src = in.const_plane_origin(0); // SYN_LAYER_PRECISION precision
    POUT *dst = out;

    // Y.
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++, dst++)
        {
            int32_t v255;
            if constexpr(std::is_same<P, int32_t>::value)
                v255 = ((*src)*max_val+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            else
                v255 = (int)((*src)*max_val + 0.5);
            if (v255 < 0)
                v255 = 0;
            else if (v255 > max_val)
                v255 = max_val;
            *dst = v255;
        }

    // U. 'nearest'
    src = in.const_plane_origin(1);
    for (int y = 0; y < h/2; y++, src += in_stride-w+in_stride)
        for (int x = 0; x < w/2; x++, src += 2, dst++)
        {
            int32_t v255;
            if constexpr(std::is_same<P, int32_t>::value)
                v255 = ((*src)*max_val+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            else
                v255 = (int)((*src)*max_val + 0.5);
            if (v255 < 0)
                v255 = 0;
            else if (v255 > max_val)
                v255 = max_val;
            *dst = v255;
        }

    // V. 'nearest'
    src = in.const_plane_origin(2);
    for (int y = 0; y < h/2; y++, src += in_stride-w+in_stride)
        for (int x = 0; x < w/2; x++, src += 2, dst++)
        {
            int32_t v255;
            if constexpr(std::is_same<P, int32_t>::value)
                v255 = ((*src)*max_val+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            else
                v255 = (int)((*src)*max_val + 0.5);
            if (v255 < 0)
                v255 = 0;
            else if (v255 > max_val)
                v255 = max_val;
            *dst = v255;
        }
}

// we 'stabilise' the yuv to (0..255) as integers prior to saving them
// incoming 444 is planar, outgoing 444 is planar.
// A non-NULL pointer (*out) is assumed to have the good size!
template <typename P, typename POUT, int max_val>
void get_raw_444_xb(POUT **param_out, struct frame_memory<P> &in)
{
    int const h = in.h;
    int const w = in.w;
    int const in_stride = in.stride;
    if (*param_out == NULL)
        *param_out = new POUT[h*w*3];
    POUT *out = *param_out;
    P *src = in.plane_origin(0); // SYN_LAYER_PRECISION precision
    POUT *dst = out;

    // Y.
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++, dst++)
        {
            int32_t v255;
            if constexpr(std::is_same<P, int32_t>::value)
                v255 = ((*src)*max_val+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            else
                v255 = (int)((*src)*max_val + 0.5);
            if (v255 < 0)
                v255 = 0;
            else if (v255 > max_val)
                v255 = max_val;
            *dst = v255;
        }

    // U. 'nearest'
    src = in.plane_origin(1);
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++, dst++)
        {
            int32_t v255;
            if constexpr(std::is_same<P, int32_t>::value)
                v255 = ((*src)*max_val+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            else
                v255 = (int)((*src)*max_val + 0.5);
            if (v255 < 0)
                v255 = 0;
            else if (v255 > max_val)
                v255 = max_val;
            *dst = v255;
        }

    // V. 'nearest'
    src = in.plane_origin(2);
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++ , dst++)
        {
            int32_t v255;
            if constexpr(std::is_same<P, int32_t>::value)
                v255 = ((*src)*max_val+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            else
                v255 = (int)((*src)*max_val + 0.5);
            if (v255 < 0)
                v255 = 0;
            else if (v255 > max_val)
                v255 = max_val;
            *dst = v255;
        }
}

// incoming 420 is planar, outgoing 444 is planar.
template <typename P, typename PIN, int max_val>
void convert_420_444_xb(frame_memory<P> &out, PIN const *in, int h, int w)
{
    out.update_to(h, w, 0, 3);

    PIN const *src = in;
    P *dst = out.plane_origin(0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            if constexpr(std::is_same<P, int32_t>::value)
                *dst = ((*src) << SYN_LAYER_PRECISION)/max_val;
            else
                *dst = ((P)*src)/max_val;

    // two output lines per input.
    P *dst0 = dst;
    P *dst1 = dst+w;
    for (int y = 0; y < h/2; y++, dst0 += w, dst1 += w)
        for (int x = 0; x < w/2; x++, src++)
        {
            P v;
            if constexpr(std::is_same<P, int32_t>::value)
                v = ((*src) << SYN_LAYER_PRECISION)/max_val;
            else
                v = ((P)*src)/max_val;
            *dst0++ = v;
            *dst0++ = v;
            *dst1++ = v;
            *dst1++ = v;
        }

    for (int y = 0; y < h/2; y++, dst0 += w, dst1 += w)
        for (int x = 0; x < w/2; x++, src++)
        {
            P v;
            if constexpr(std::is_same<P, int32_t>::value)
                v = ((*src) << SYN_LAYER_PRECISION)/max_val;
            else
                v = ((P)*src)/max_val;
            *dst0++ = v;
            *dst0++ = v;
            *dst1++ = v;
            *dst1++ = v;
        }
}

// frame_420: byte, planar, seeks to frame position.
void dump_yuv420_8b(unsigned char *frame_420, int h, int w, FILE *fout, int frame_idx)
{
    fseek(fout, frame_idx*h*w*3*sizeof(frame_420[0])/2, SEEK_SET);
    fwrite(frame_420, sizeof(frame_420[0]), h*w*3/2, fout);
    fflush(fout);
}

void dump_yuv420_10b(unsigned short *frame_420, int h, int w, FILE *fout, int frame_idx)
{
    fseek(fout, frame_idx*h*w*3*sizeof(frame_420[0])/2, SEEK_SET);
    fwrite(frame_420, sizeof(frame_420[0]), h*w*3/2, fout);
    fflush(fout);
}

// frame_444: byte, planar, seeks to frame position.
void dump_yuv444_8b(unsigned char *frame_444, int h, int w, FILE *fout, int frame_idx)
{
    fseek(fout, frame_idx*h*w*3*sizeof(frame_444[0]), SEEK_SET);
    fwrite(frame_444, sizeof(frame_444[0]), h*w*3, fout);
    fflush(fout);
}

// frame_444: byte, planar, seeks to frame position.
void dump_yuv444_10b(unsigned short *frame_444, int h, int w, FILE *fout, int frame_idx)
{
    fseek(fout, frame_idx*h*w*3*sizeof(frame_444[0]), SEEK_SET);
    fwrite(frame_444, sizeof(frame_444[0]), h*w*3, fout);
    fflush(fout);
}


// incoming 444 is planar, outgoing 444 is planar.
template <typename P, typename PIN, int max_val>
void store_444_xb(frame_memory<P> &out, PIN *in, int h, int w)
{
    out.update_to(h, w, 0, 3);

    PIN *src = in;
    P *dst = out.plane_origin(0);
    // Y
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            if constexpr(std::is_same<P, int32_t>::value)
                *dst = ((*src) << SYN_LAYER_PRECISION)/max_val;
            else
                *dst = ((P)*src)/max_val;

    // U
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            if constexpr(std::is_same<P, int32_t>::value)
                *dst = ((*src) << SYN_LAYER_PRECISION)/max_val;
            else
                *dst = ((P)*src)/max_val;

    // V
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            if constexpr(std::is_same<P, int32_t>::value)
                *dst = ((*src) << SYN_LAYER_PRECISION)/max_val;
            else
                *dst = ((P)*src)/max_val;

}

// final b-prediction
// add two pre-multiplied preds, multiply by alpha, add residue.
template <typename P>
void bpred(struct frame_memory<P> &bpred_result, struct frame_memory<P> const &residual, struct frame_memory<P> const &pred0, struct frame_memory<P> const &pred1, int raw_gainidx)
{
    const auto time_bpred_start = std::chrono::steady_clock::now();
    int const h = residual.h;
    int const w = residual.w;
    P const *raw = residual.const_origin();
    int raw_stride = residual.stride;
    int raw_plane_stride = residual.plane_stride;

    bpred_result.update_to(h, w, 0, 3);

    P const *src0 = pred0.const_origin();
    P const *src1 = pred1.const_origin();
    int const src_plane_stride = pred0.plane_stride;

    if (pred0.plane_stride != pred1.plane_stride || pred0.stride != pred0.w)
    {
        printf("internal error\n");
        printf("p0 stride %d width %d; p1 %d %d\n", pred0.stride, pred0.w, pred1.stride, pred1.w);
        exit(1);
    }

    P *dst = bpred_result.origin(); // first plane.
    int const raw_gain_offset = raw_gainidx * raw_plane_stride;

    for (int y = 0; y < h; y++, raw += raw_stride-w)
    for (int x = 0; x < w; x++, raw++, dst++, src0++, src1++)
    {
        if constexpr(std::is_same<P, int32_t>::value)
        {
            P gain = raw[raw_gain_offset]+(1<<(SYN_LAYER_PRECISION-1));
            if (gain < 0)
                gain = 0;
            else if (gain > (1<<SYN_LAYER_PRECISION))
                gain = (1<<SYN_LAYER_PRECISION);

            // clamp.
            for (int c = 0; c < 3; c++)
            {
                P res = (((src0[c*src_plane_stride]+src1[c*src_plane_stride])*gain)>>SYN_LAYER_PRECISION) + raw[c*raw_plane_stride];
                if (res < 0)
                    res = 0;
                else if (res > (1<<SYN_LAYER_PRECISION))
                    res = (1<<SYN_LAYER_PRECISION);
                dst[c*src_plane_stride] = res;
            }
        }
        else
        {
            P alpha = raw[raw_gain_offset]+0.5;
            if (alpha < 0)
                alpha = 0;
            else if (alpha > 1.0)
                alpha = 1.0;

            // clamp.
            for (int c = 0; c < 3; c++)
            {
                P res = (src0[c*src_plane_stride]+src1[c*src_plane_stride])*alpha + raw[c*raw_plane_stride];
                if (res < 0)
                    res = 0;
                else if (res > 1.0)
                    res = 1.0;
                dst[c*src_plane_stride] = res;
            }
        }
    }

    const auto time_bpred_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_bpred = (time_bpred_end-time_bpred_start);
    time_bpred_seconds += (float)elapsed_bpred.count();
}

// returns a yuv444 float prediction, planar.
// refx_444: float, planar.
// residual: rgb + alpha
// motion:   xy (P) or xy+xy+beta (B) -- CAN BE REDUCED RESOLUTION.
template <typename P>
void process_inter(struct frame_memory<P> &pred, struct frame_memory<P> const &residual, frame_memory<P> /*const*/ &motion, struct frame_memory<P> const *ref0_444, struct frame_memory<P> const *ref1_444,
                   int global_flow[2][2],
                   struct frame_memory<P> &warp0, struct frame_memory <P> &warp1,
                   int ntaps, int motion_blksz_shift, int motion_q)
{
    // work out the 'motion block size'.
    if (ref1_444 == NULL)
    {
        // P
        // multiplies by alpha, adds residue.
        warp_ntaps<P, float>(pred, residual, motion, *ref0_444, global_flow[0], 0, 3, true, ntaps, motion_blksz_shift, motion_q);
    }
    else
    {
        // B
        // want to multiply by beta, do not add residue.
        warp_ntaps<P, float>(warp0, residual, motion, *ref0_444, global_flow[0], 0, 4, false, ntaps, motion_blksz_shift, motion_q);
        warp_ntaps<P, float>(warp1, residual, motion, *ref1_444, global_flow[1], 2, -4, false, ntaps, motion_blksz_shift, motion_q);

        // want to add warp0, warp1, multiply by alpha, and add residue.
        bpred(pred, residual, warp0, warp1, 3);
    }
}

template <typename P>
struct frame_memory<P> *get_444_ref_frame(std::vector<struct frame_memory<P> *> &frame_444_buffer)
{
    if (frame_444_buffer.size() == 0)
        return new struct frame_memory<P>[1];

    auto result = frame_444_buffer.back();
    frame_444_buffer.pop_back();
    return result;
}

template <typename P>
void put_444_ref_frame(std::vector<struct frame_memory<P> *> &frame_444_buffer, struct frame_memory<P> *frame)
{
    frame_444_buffer.emplace_back(frame);
}

#if defined(CCDECAPI_CPU)
int cc_decode_cpu(std::string &bitstream_filename, std::string &out_filename, int output_bitdepth, int output_chroma_format, int motion_q, int verbosity)
#elif defined(CCDECAPI_AVX2_OPTIONAL)
int cc_decode_avx2_optional(std::string &bitstream_filename, std::string &out_filename, int output_bitdepth, int output_chroma_format, bool use_avx2, int motion_q, int verbosity)
#elif defined(CCDECAPI_AVX2)
int cc_decode_avx2(std::string &bitstream_filename, std::string &out_filename, int output_bitdepth, int output_chroma_format, int motion_q, int verbosity)
#else
#error must have one of CCDECAPI_CPU, CCDECAPI_AVX2 or CCDECAPI_AVX2_OPTIONAL defined.
#endif
{
    if (bitstream_filename == "")
    {
        printf("must specify a bitstream\n");
        return 1;
    }

    // ensure motion_q is a power of 2. Limit to 2^10.
    int motion_q_shift = 0;
    while ((1<<motion_q_shift) < motion_q && motion_q_shift < 10)
        motion_q_shift += 1;
    if ((1<<motion_q_shift) != motion_q)
    {
        printf("motion_accuracy: must be power of 2, 2^10 max.  Received %d.\n", motion_q);
        exit(1);
    }

    const auto time_all_start = std::chrono::steady_clock::now();

    cc_bs bs;
    if (!bs.open(bitstream_filename, verbosity))
    {
        printf("cannot open %s for reading\n", bitstream_filename.c_str());
        return 1;
    }

    struct cc_bs_gop_header &gop_header = bs.m_gop_header;
    std::vector<struct frame_memory<SYN_INT_FLOAT> *> frame_444_buffer; // frame data that can be reused.
    std::vector<struct frame_memory<SYN_INT_FLOAT> *> ref_frames_444; // reference frames in display_order positions.  NULL for unseen.
    frame_memory<SYN_INT_FLOAT> warp0; // for inter-processing
    frame_memory<SYN_INT_FLOAT> warp1; // for inter-processing

    // video 420, 444, 8b, 10b conversion result for video.
    // just a single temporary.
    // we 'bounce through' it going from 444 fixed point original to yuv out and back.
    unsigned char  *frame_4XX_8b = NULL;
    unsigned short *frame_4XX_10b = NULL;

    struct frame_memory<SYN_INT_FLOAT> raw_cc_outputs[2]; // residue, motion.

    FILE *fout = NULL;
    if (out_filename != "")
    {
        fout = fopen(out_filename.c_str(), "wb");
        if (fout == NULL)
        {
            printf("Cannot open %s for writing\n", out_filename.c_str());
            return 1;
        }
    }

#if defined(CCDECAPI_AVX2_OPTIONAL)
    struct cc_frame_decoder frame_decoder(gop_header, output_bitdepth, output_chroma_format, use_avx2, verbosity);
#else
    struct cc_frame_decoder frame_decoder(gop_header, output_bitdepth, output_chroma_format, verbosity);
#endif

    cc_bs_frame *prev_coded_I_frame_symbols = NULL;
    cc_bs_frame *prev_coded_frame_symbols = NULL;
    for (int frame_coding_idx = 0; true; /*gop_header.intra_period;*/ frame_coding_idx++)
    {
        // raw_cc_output either
        // [3] (I: [residue])
        // [6] (P: [residue+alpha] [xy])
        // [9] (B: [residue+alpha] [xy+xy+beta])

        cc_bs_frame *frame_symbols = bs.decode_frame(prev_coded_I_frame_symbols, prev_coded_frame_symbols, verbosity);
        if (frame_symbols == NULL)
        {
            printf("EOF at frame %d\n", frame_coding_idx);
            break;
        }

        struct cc_bs_frame_header &frame_header = frame_symbols->m_frame_header;
        // Add NULL placeholders for unseen frames prior to display_index.
        while (frame_header.display_index >= (int)ref_frames_444.size())
            ref_frames_444.push_back(NULL);

        // raw frame data. !!! COPIED
        int idx = 0;
        for (auto &cc: frame_symbols->m_coolchics)
        {
            // COPY FOR NOW TO GET OTHER LOGIC RIGHT
            struct frame_memory<SYN_INT_FLOAT> *out = frame_decoder.decode_frame(cc);
            if (out == NULL)
            {
                printf("decoding failed!\n");
                exit(1);
            }
            raw_cc_outputs[idx].copy_from(*out);
            idx++;
        }

        struct frame_memory<SYN_INT_FLOAT> *frame_444p; // direct intra.
        struct frame_memory<SYN_INT_FLOAT> frame_444;   // inter-created.

        // intra/inter processing.
        if (frame_header.frame_type == 'I')
        {
            // I
            frame_444p = &raw_cc_outputs[0];
        }
        else
        {
            // search for references.
            struct frame_memory<SYN_INT_FLOAT> *ref_prev = NULL;
            struct frame_memory<SYN_INT_FLOAT> *ref_next = NULL;
            // find prev for P and B.
            for (int idx = frame_header.display_index-1; idx >= 0; idx--)
                if (ref_frames_444[idx] != NULL)
                {
                    ref_prev = ref_frames_444[idx];
                    break;
                }
            if (ref_prev == NULL)
            {
                printf("error finding left reference for display idx %d\n", frame_header.display_index);
                return 1;
            }

            // find next if B.
            if (frame_header.frame_type == 'B')
            {
                for (int idx = frame_header.display_index+1; idx < (int)ref_frames_444.size(); idx++)
                    if (ref_frames_444[idx] != NULL)
                    {
                        ref_next = ref_frames_444[idx];
                        break;
                    }
                if (ref_next == NULL)
                {
                    printf("error finding right reference for display idx %d\n", frame_header.display_index);
                    return 1;
                }
            }
            process_inter(frame_444, raw_cc_outputs[0], raw_cc_outputs[1], ref_prev, ref_next, frame_header.global_flow, warp0, warp1, frame_header.warp_filter_length, frame_symbols->m_coolchics[1].n_leading_zero_feature_layers, motion_q_shift);
            frame_444p = &frame_444;
        }

        // THRU YUV 420
        struct frame_memory<SYN_INT_FLOAT> *result_444 = get_444_ref_frame(frame_444_buffer);
        // 0 = RGB
        if (frame_decoder.m_frame_data_type != 0) {
            if (frame_decoder.m_output_chroma_format == 420)
            {
                if (frame_decoder.m_output_bitdepth == 8)
                {
                    convert_444_420_xb<SYN_INT_FLOAT, unsigned char, 255>(&frame_4XX_8b, *frame_444p);
                    if (out_filename != "")
                        dump_yuv420_8b(frame_4XX_8b, gop_header.img_h, gop_header.img_w, fout, frame_header.display_index);
                    convert_420_444_xb<SYN_INT_FLOAT, unsigned char, 255>(*result_444, frame_4XX_8b, gop_header.img_h, gop_header.img_w);
                }
                else if (frame_decoder.m_output_bitdepth == 10)
                {
                    convert_444_420_xb<SYN_INT_FLOAT, unsigned short, 1023>(&frame_4XX_10b, *frame_444p);
                    if (out_filename != "")
                        dump_yuv420_10b(frame_4XX_10b, gop_header.img_h, gop_header.img_w, fout, frame_header.display_index);
                    convert_420_444_xb<SYN_INT_FLOAT, unsigned short, 1023>(*result_444, frame_4XX_10b, gop_header.img_h, gop_header.img_w);
                }
                else
                {
                    printf("Unknown YUV bitdepth %d. Should be 8 or 10.\n", frame_decoder.m_output_bitdepth);
                    exit(1);
                }
            }
            // THRU YUV 444
            else if (frame_decoder.m_output_chroma_format == 444)
            {
                if (frame_decoder.m_output_bitdepth == 8)
                {
                    get_raw_444_xb<SYN_INT_FLOAT, unsigned char, 255>(&frame_4XX_8b, *frame_444p);
                    if (out_filename != "")
                        dump_yuv444_8b(frame_4XX_8b, gop_header.img_h, gop_header.img_w, fout, frame_header.display_index);
                    store_444_xb<SYN_INT_FLOAT, unsigned char, 255>(*result_444, frame_4XX_8b, gop_header.img_h, gop_header.img_w);
                }
                else if (frame_decoder.m_output_bitdepth == 10)
                {
                    get_raw_444_xb<SYN_INT_FLOAT, unsigned short, 1023>(&frame_4XX_10b, *frame_444p);
                    if (out_filename != "")
                        dump_yuv444_10b(frame_4XX_10b, gop_header.img_h, gop_header.img_w, fout, frame_header.display_index);
                    store_444_xb<SYN_INT_FLOAT, unsigned short, 1023>(*result_444, frame_4XX_10b, gop_header.img_h, gop_header.img_w);
                }
                else
                {
                    printf("Unknown YUV bitdepth %d. Should be 8 or 10.\n", frame_decoder.m_output_bitdepth);
                    exit(1);
                }
            }
        }
        else
        {
            // rgb
            if (out_filename != "")
                ppm_out(3, frame_decoder.m_output_bitdepth, *frame_444p, fout);
        }
        ref_frames_444[frame_header.display_index] = result_444;

        // !!! temporary
        // scan to the left of display_index to see what we can remove.
        // we are assuming an incoming display index frame can use, as a reference, only the
        // closest frame present to the left.
        // the cleanup is triggered by the display_index just added.
        // if it is at the rightmost end of a block of refs, with only NULLs to the left of that
        // block, we remove all elements in the block except the display_index just added.
        int block_start = frame_header.display_index-1;
        int block_end = frame_header.display_index+1;
        // start of block?
        while (block_start >= 0 && ref_frames_444[block_start] != NULL)
            block_start -= 1;
        block_start += 1; // the first non-NULL element in the block.

        // search to the right for the rightmost limit.
        while (block_end < (int)ref_frames_444.size() && ref_frames_444[block_end] != NULL)
            block_end += 1;
        block_end -= 1; // the last non-NULL element in the block.

        // the block contains more than 1 element and there are only NULLs to the left of block_start?
        if (block_start < block_end) // at least 2 elements in the 'block'
        {
            int more_refs = false; // more references found to the left of the block.
            for (int idx = block_start-1; idx >= 0; idx -= 1)
            {
                if (ref_frames_444[idx] != NULL)
                {
                    more_refs = true;
                    break;
                }
            }
            if (more_refs)
            {
                // cannot remove leftmost elements in block, it is not the 1st block.
            }
            else
            {
                // remove all elements in block except the rightmost.
                for (int idx = block_start; idx < block_end; idx++)
                {
                    // printf("display %d: frame %d for reuse\n", frame_header.display_index, idx);
                    put_444_ref_frame(frame_444_buffer, ref_frames_444[idx]);
                    ref_frames_444[idx] = NULL;
                }
            }
        }

        if (frame_symbols->m_frame_header.frame_type == 'I')
        {
            delete prev_coded_I_frame_symbols;
            prev_coded_I_frame_symbols = frame_symbols;
            delete prev_coded_frame_symbols;
            prev_coded_frame_symbols = NULL;
        }
        else
        {
            delete prev_coded_frame_symbols;
            prev_coded_frame_symbols = frame_symbols;
        }

        const auto time_all_end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed_all = (time_all_end-time_all_start);
        time_all_seconds = (float)elapsed_all.count();

        if (verbosity >= 1)
            printf("time: arm %g ups %g syn %g blend %g warp %g (coeffgen %g coeffget %g) bpred %g all %g\n",
                    time_arm_seconds, time_ups_seconds, time_syn_seconds, time_blend_seconds, time_warp_seconds, time_warp_coeffgen_seconds, time_warp_coeffget_seconds, time_bpred_seconds, time_all_seconds);
        fflush(stdout);
    }

    // !!! free buffers.
    delete prev_coded_frame_symbols;
    delete prev_coded_I_frame_symbols;

    if (fout != NULL)
    {
        fclose(fout);
        printf("%s created\n", out_filename.c_str());
        fflush(stdout);
    }

    return 0;
}


#ifdef CCDEC_EXE

// A main() and associated parameters for standalone executable.
#ifndef CCDECAPI_AVX2_OPTIONAL
// we are expecting 'optional' -- we have a run-time parameter for choosing cpu or avx2 or auto.
#error CCDEC_EXE needs CCDECAPI_AVX2_OPTIONAL
no.
#endif

char const *param_bitstream = "--input=";
char const *param_out       = "--output=";
char const *param_bitdepth  = "--output_bitdepth=";
char const *param_chroma    = "--output_chroma_format=";
char const *param_cpu       = "--cpu";
char const *param_auto      = "--auto";
char const *param_avx2      = "--avx2";
char const *param_motion_q  = "--motion_accuracy=";
char const *param_v         = "--v=";

void usage(char const *msg = NULL)
{
    if (msg != NULL)
        printf("%s\n", msg);
    printf("Usage:\n");
    printf("    %s: .bin to decode\n", param_bitstream);
    printf("    [%s]: reconstruction if desired: ppm (image) or yuv420 (video) only\n", param_out);
    printf("    [%s]: output bitdepth 0 => take from bitstream\n", param_bitdepth);
    printf("    [%s]: output chroma subsampling 420 or 444 for yuv; 0 => take from bitstream\n", param_chroma);
    printf("    [%s|%s|%s]: optimized instruction set with which to decode\n", param_cpu, param_avx2, param_auto);
    printf("    [%s]: motion_accuracy: n for 1/n (default 64)\n", param_motion_q);
    printf("    [%s]: verbosity\n", param_v);
    exit(msg != NULL);
}

int main(int argc, const char* argv[])
{

    std::string bitstream_filename;
    std::string out = "out";
    int output_bitdepth = 0;
    int output_chroma_format = 0;
    bool explicit_instruction_set = false;
    bool use_avx2 = false;
    bool use_auto = false;
    int motion_q = 64;
    int  verbosity = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strncmp(argv[i], param_bitstream, strlen(param_bitstream)) == 0)
            bitstream_filename = argv[i]+strlen(param_bitstream);
        else if (strncmp(argv[i], param_out, strlen(param_out)) == 0)
            out = argv[i]+strlen(param_out);
        else if (strncmp(argv[i], param_bitdepth, strlen(param_bitdepth)) == 0)
            output_bitdepth = atoi(argv[i]+strlen(param_bitdepth));
        else if (strncmp(argv[i], param_chroma, strlen(param_chroma)) == 0)
            output_chroma_format = atoi(argv[i]+strlen(param_chroma));
        else if (strncmp(argv[i], param_avx2, strlen(param_avx2)) == 0
                 || strncmp(argv[i], param_cpu, strlen(param_cpu)) == 0
                 || strncmp(argv[i], param_auto, strlen(param_auto)) == 0)
        {
            if (explicit_instruction_set)
            {
                printf("%s: only a single --auto, --avx2 or --cpu\n", argv[i]);
                exit(1);
            }
            explicit_instruction_set = true;
            use_avx2 = strncmp(argv[i], param_avx2, strlen(param_avx2)) == 0;
            use_auto = strncmp(argv[i], param_auto, strlen(param_auto)) == 0;
        }
        else if (strncmp(argv[i], param_motion_q, strlen(param_motion_q)) == 0)
        {
            motion_q = atoi(argv[i]+strlen(param_motion_q));
        }
        else if (strncmp(argv[i], param_v, strlen(param_v)) == 0)
        {
            verbosity = atoi(argv[i]+strlen(param_v));
        }
        else
        {
            std::string error_message = "unknown parameter ";
            error_message += argv[i];
            usage(error_message.c_str());
        }
    }

    if (bitstream_filename == "")
        usage("must specify a bitstream");

    if (!explicit_instruction_set)
        use_auto = true;
    if (use_auto)
    {
        if (__builtin_cpu_supports("avx2"))
        {
            use_avx2 = true;
        }
    }
    int result = cc_decode_avx2_optional(bitstream_filename, out, output_bitdepth, output_chroma_format, use_avx2, motion_q, verbosity);
    return result;
}
#endif
