
/*
 * USED WITH
 * #define CCDECAPI_CPU
 * or
 * #define CCDECAPI_AVX2
 */

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

float time_bac_seconds = 0.0;
float time_arm_seconds = 0.0;
float time_ups_seconds = 0.0;
float time_syn_seconds = 0.0;
float time_warp_seconds = 0.0;
float time_bpred_seconds = 0.0;
float time_all_seconds = 0.0;

// only use nc == 3
void ppm_out(int nc, int pixel_depth, struct frame_memory &in_info, char const *outname)
{
    int const h = in_info.h;
    int const w = in_info.w;
    int const stride = in_info.stride;
    int const plane_stride = in_info.plane_stride;
    int32_t *ins = in_info.origin();
    FILE *fout = fopen(outname, "wb");
    if (fout == NULL)
    {
        printf("Cannot open %s for writing.\n", outname);
        exit(1);
    }

    if (nc == 3)
    {
        fprintf(fout, "P6\n");
        fprintf(fout, "%d %d\n", w, h);
        fprintf(fout, "255\n");
        int32_t *inR = ins;
        int32_t *inG = inR+plane_stride;
        int32_t *inB = inG+plane_stride;
        for (int y = 0; y < h; y++, inR += stride-w, inG += stride-w, inB += stride-w)
            for (int x = 0; x < w; x++)
            {
                // precision is SYN_LAYER_PRECISION
                int r = ((*inR++)*255+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
                int g = ((*inG++)*255+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
                int b = ((*inB++)*255+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
                if (r < 0) r = 0;
                if (g < 0) g = 0;
                if (b < 0) b = 0;
                if (r > 255) r = 255;
                if (g > 255) g = 255;
                if (b > 255) b = 255;
                unsigned char pix[3];
                pix[0] = r;
                pix[1] = g;
                pix[2] = b;
                fwrite(pix, 3, 1, fout);
            }
    }

    fclose(fout);
}

// we 'stabilise' the yuv to (0..255) as integers, as well as uv subsample.
// incoming 444 is planar, outgoing 420 is planar.
unsigned char *convert_444_420_8b(struct frame_memory &in)
{
    int const h = in.h;
    int const w = in.w;
    int const in_stride = in.stride;
    unsigned char *out = new unsigned char[h*w*3/2];
    int32_t *src = in.plane_origin(0); // SYN_LAYER_PRECISION precision
    unsigned char *dst = out;

    // Y.
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++, dst++)
        {
            int32_t v255 = ((*src)*255+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            if (v255 < 0)
                v255 = 0;
            else if (v255 > 255)
                v255 = 255;
            *dst = v255;
        }

    // U. 'nearest'
    src = in.plane_origin(1);
    for (int y = 0; y < h/2; y++, src += in_stride-w+in_stride)
        for (int x = 0; x < w/2; x++, src += 2, dst++)
        {
            int32_t v255 = ((*src)*255+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            if (v255 < 0)
                v255 = 0;
            else if (v255 > 255)
                v255 = 255;
            *dst = v255;
        }

    // V. 'nearest'
    src = in.plane_origin(2);
    for (int y = 0; y < h/2; y++, src += in_stride-w+in_stride)
        for (int x = 0; x < w/2; x++, src += 2, dst++)
        {
            int32_t v255 = ((*src)*255+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            if (v255 < 0)
                v255 = 0;
            else if (v255 > 255)
                v255 = 255;
            *dst = v255;
        }

    return out;
}

// we 'stabilise' the yuv to (0..1023) as integers, as well as uv subsample.
// incoming 444 is planar, outgoing 420 is planar.
unsigned short *convert_444_420_10b(struct frame_memory &in)
{
    int const h = in.h;
    int const w = in.w;
    int const in_stride = in.stride;
    unsigned short *out = new unsigned short[h*w*3/2];
    int32_t *src = in.plane_origin(0); // SYN_LAYER_PRECISION precision
    unsigned short *dst = out;

    // Y.
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++, dst++)
        {
            int32_t v1023 = ((*src)*1023+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            if (v1023 < 0)
                v1023 = 0;
            else if (v1023 > 1023)
                v1023 = 1023;
            *dst = v1023;
        }

    // U. 'nearest'
    src = in.plane_origin(1);
    for (int y = 0; y < h/2; y++, src += in_stride-w+in_stride)
        for (int x = 0; x < w/2; x++, src += 2, dst++)
        {
            int32_t v1023 = ((*src)*1023+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            if (v1023 < 0)
                v1023 = 0;
            else if (v1023 > 1023)
                v1023 = 1023;
            *dst = v1023;
        }

    // V. 'nearest'
    src = in.plane_origin(2);
    for (int y = 0; y < h/2; y++, src += in_stride-w+in_stride)
        for (int x = 0; x < w/2; x++, src += 2, dst++)
        {
            int32_t v1023 = ((*src)*1023+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            if (v1023 < 0)
                v1023 = 0;
            else if (v1023 > 1023)
                v1023 = 1023;
            *dst = v1023;
        }

    return out;
}

// we 'stabilise' the yuv to (0..255) as integers prior to saving them
// incoming 444 is planar, outgoing 444 is planar.
unsigned char *get_raw_444_8b(struct frame_memory &in)
{
    int const h = in.h;
    int const w = in.w;
    int const in_stride = in.stride;
    unsigned char *out = new unsigned char[h*w*3];
    int32_t *src = in.plane_origin(0); // SYN_LAYER_PRECISION precision
    unsigned char *dst = out;

    // Y.
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++, dst++)
        {
            int32_t v255 = ((*src)*255+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            if (v255 < 0)
                v255 = 0;
            else if (v255 > 255)
                v255 = 255;
            *dst = v255;
        }

    // U. 'nearest'
    src = in.plane_origin(1);
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++, dst++)
        {
            int32_t v255 = ((*src)*255+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            if (v255 < 0)
                v255 = 0;
            else if (v255 > 255)
                v255 = 255;
            *dst = v255;
        }

    // V. 'nearest'
    src = in.plane_origin(2);
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++ , dst++)
        {
            int32_t v255 = ((*src)*255+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            if (v255 < 0)
                v255 = 0;
            else if (v255 > 255)
                v255 = 255;
            *dst = v255;
        }

    return out;
}

// we 'stabilise' the yuv to (0..1023) as integers prior to saving them
// incoming 444 is planar, outgoing 444 is planar.
unsigned short *get_raw_444_10b(struct frame_memory &in)
{
    int const h = in.h;
    int const w = in.w;
    int const in_stride = in.stride;
    unsigned short *out = new unsigned short[h*w*3];
    int32_t *src = in.plane_origin(0); // SYN_LAYER_PRECISION precision
    unsigned short *dst = out;

    // Y.
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++, dst++)
        {
            int32_t v1023 = ((*src)*1023+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            if (v1023 < 0)
                v1023 = 0;
            else if (v1023 > 1023)
                v1023 = 1023;
            *dst = v1023;
        }

    // U. 'nearest'
    src = in.plane_origin(1);
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++, dst++)
        {
            int32_t v1023 = ((*src)*1023+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            if (v1023 < 0)
                v1023 = 0;
            else if (v1023 > 1023)
                v1023 = 1023;
            *dst = v1023;
        }

    // V. 'nearest'
    src = in.plane_origin(2);
    for (int y = 0; y < h; y++, src += in_stride-w)
        for (int x = 0; x < w; x++, src++, dst++)
        {
            int32_t v1023 = ((*src)*1023+(1<<(SYN_LAYER_PRECISION-1))) >> SYN_LAYER_PRECISION;
            if (v1023 < 0)
                v1023 = 0;
            else if (v1023 > 1023)
                v1023 = 1023;
            *dst = v1023;
        }

    return out;
}





// incoming 420 is planar, outgoing 444 is planar.
void convert_420_444_8b(frame_memory &out, unsigned char *in, int h, int w)
{
    out.update_to(h, w, 3, 0);

    unsigned char *src = in;
    int32_t *dst = out.plane_origin(0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            *dst = ((*src) << SYN_LAYER_PRECISION)/255;

    // two output lines per input.
    int32_t *dst0 = dst;
    int32_t *dst1 = dst+w;
    for (int y = 0; y < h/2; y++, dst0 += w, dst1 += w)
        for (int x = 0; x < w/2; x++, src++)
        {
            int32_t v = ((*src) << SYN_LAYER_PRECISION)/255;
            *dst0++ = v;
            *dst0++ = v;
            *dst1++ = v;
            *dst1++ = v;
        }

    for (int y = 0; y < h/2; y++, dst0 += w, dst1 += w)
        for (int x = 0; x < w/2; x++, src++)
        {
            int32_t v = ((*src) << SYN_LAYER_PRECISION)/255;
            *dst0++ = v;
            *dst0++ = v;
            *dst1++ = v;
            *dst1++ = v;
        }
}

// incoming 420 is planar, outgoing 444 is planar.
void convert_420_444_10b(frame_memory &out, unsigned short *in, int h, int w)
{
    out.update_to(h, w, 3, 0);

    unsigned short *src = in;
    int32_t *dst = out.plane_origin(0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            *dst = ((*src) << SYN_LAYER_PRECISION)/1023;

    // two output lines per input.
    int32_t *dst0 = dst;
    int32_t *dst1 = dst+w;
    for (int y = 0; y < h/2; y++, dst0 += w, dst1 += w)
        for (int x = 0; x < w/2; x++, src++)
        {
            int32_t v = ((*src) << SYN_LAYER_PRECISION)/1023;
            *dst0++ = v;
            *dst0++ = v;
            *dst1++ = v;
            *dst1++ = v;
        }

    for (int y = 0; y < h/2; y++, dst0 += w, dst1 += w)
        for (int x = 0; x < w/2; x++, src++)
        {
            int32_t v = ((*src) << SYN_LAYER_PRECISION)/1023;
            *dst0++ = v;
            *dst0++ = v;
            *dst1++ = v;
            *dst1++ = v;
        }
}


// frame_420: byte, planar, seeks to frame position.
void dump_yuv420_8b(unsigned char *frame_420, int h, int w, FILE *fout, int frame_idx)
{
    fseek(fout, frame_idx*h*w*3/2, SEEK_SET);
    fwrite(frame_420, sizeof(frame_420[0]), h*w*3/2, fout);
    fflush(fout);
}

void dump_yuv420_10b(unsigned short *frame_420, int h, int w, FILE *fout, int frame_idx)
{
    fseek(fout, frame_idx*h*w*3/2, SEEK_SET);
    fwrite(frame_420, sizeof(frame_420[0]), h*w*3/2, fout);
    fflush(fout);
}

// frame_444: byte, planar, seeks to frame position.
void dump_yuv444_8b(unsigned char *frame_444, int h, int w, FILE *fout, int frame_idx)
{
    fseek(fout, frame_idx*h*w*3, SEEK_SET);
    fwrite(frame_444, sizeof(frame_444[0]), h*w*3, fout);
    fflush(fout);
}

// frame_444: byte, planar, seeks to frame position.
void dump_yuv444_10b(unsigned short *frame_444, int h, int w, FILE *fout, int frame_idx)
{
    fseek(fout, frame_idx*h*w*3, SEEK_SET);
    fwrite(frame_444, sizeof(frame_444[0]), h*w*3, fout);
    fflush(fout);
}


// incoming 444 is planar, outgoing 444 is planar.
void store_444_8b(frame_memory &out, unsigned char *in, int h, int w)
{
    out.update_to(h, w, 3, 0);

    unsigned char *src = in;
    int32_t *dst = out.plane_origin(0);
    // Y
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            *dst = ((*src) << SYN_LAYER_PRECISION)/255;

    // U
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            *dst = ((*src) << SYN_LAYER_PRECISION)/255;

    // V
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            *dst = ((*src) << SYN_LAYER_PRECISION)/255;

}

// incoming 444 is planar, outgoing 444 is planar.
void store_444_10b(frame_memory &out, unsigned short *in, int h, int w)
{
    out.update_to(h, w, 3, 0);

    unsigned short *src = in;
    int32_t *dst = out.plane_origin(0);
    // Y
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            *dst = ((*src) << SYN_LAYER_PRECISION)/1023;

    // U
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            *dst = ((*src) << SYN_LAYER_PRECISION)/1023;

    // V
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++, src++, dst++)
            *dst = ((*src) << SYN_LAYER_PRECISION)/1023;
}


// returns a yuv444 SYN_LAYER_PRECISION precision, warping a reference, apply a gain
// bilinear sampling.
// ref: float, planar.
void warp(struct frame_memory &warp_result, struct frame_memory &raw_info, int raw_info_depth, struct frame_memory &ref, int raw_xyidx, int raw_gainidx, int flo_gain, bool add_residue)
{
    const auto time_warp_start = std::chrono::steady_clock::now();

    warp_result.update_to(ref.h, ref.w, 3, 0);

    int32_t *src = ref.origin();
    int const src_stride = ref.stride;
    int const src_plane_stride = ref.plane_stride;
    int32_t *raw = raw_info.origin();
    int const raw_stride = raw_info.stride;
    int const raw_plane_stride = raw_info.plane_stride;
    int const h = raw_info.h;
    int const w = raw_info.w;

    int32_t *out = warp_result.origin();
    int const out_plane_stride = warp_result.plane_stride;

    int32_t *dst = out; // scans through first plane.

    for (int y = 0; y < h; y++, raw += raw_stride-w)
    for (int x = 0; x < w; x++, raw++, dst++)
    {
        int32_t px = (raw[(raw_xyidx+0)*raw_plane_stride]*flo_gain)+(x<<SYN_LAYER_PRECISION); // flo_gain is 0 or 1.
        int32_t py = (raw[(raw_xyidx+1)*raw_plane_stride]*flo_gain)+(y<<SYN_LAYER_PRECISION); // flo_gain is 0 or 1.

        int32_t basex0 = px < 0 ? ((px-((1<<SYN_LAYER_PRECISION)-1)) >> SYN_LAYER_PRECISION) : (px >> SYN_LAYER_PRECISION);
        int32_t basex1 = basex0+1;
        int32_t dx = px-(basex0<<SYN_LAYER_PRECISION);
        if (dx < 0) {printf("what dx %d %d\n", px, basex0);exit(1); }
        if (basex0 < 0)
        {
            basex0 = 0;
            basex1 = basex0;
            dx = 0;
        }
        else if (basex0 >= w-1)
        {
            basex0 = w-1;
            basex1 = basex0;
            dx = 0;
        }
        int32_t basey0 = py < 0 ? ((py-((1<<SYN_LAYER_PRECISION)-1)) >> SYN_LAYER_PRECISION) : (py >> SYN_LAYER_PRECISION);
        int32_t basey1 = basey0+1;
        int32_t dy = py-(basey0<<SYN_LAYER_PRECISION);
        if (dy < 0) {printf("what dy %d %d\n", py, basey0);exit(1); }
        if (basey0 < 0)
        {
            basey0 = 0;
            basey1 = basey0;
            dy = 0;
        }
        else if (basey0 >= h-1)
        {
            basey0 = h-1;
            basey1 = basey0;
            dy = 0;
        }

        int32_t gain;
        if (raw_gainidx < 0)
            gain = raw[-raw_gainidx*raw_plane_stride]+(1<<(SYN_LAYER_PRECISION-1));
        else
            gain = raw[raw_gainidx*raw_plane_stride]+(1<<(SYN_LAYER_PRECISION-1));
        if (gain < 0)
            gain = 0;
        else if (gain > (1<<SYN_LAYER_PRECISION))
            gain = (1<<SYN_LAYER_PRECISION);
        if (raw_gainidx < 0)
            gain = (1<<SYN_LAYER_PRECISION)-gain;

        int32_t *A = &src[((basey0)*src_stride+basex0)];
        int32_t *B = &src[((basey0)*src_stride+basex1)];
        int32_t *C = &src[((basey1)*src_stride+basex0)];
        int32_t *D = &src[((basey1)*src_stride+basex1)];

        int32_t h0 = A[0*src_plane_stride]+(((B[0*src_plane_stride]-A[0*src_plane_stride])*dx) >> SYN_LAYER_PRECISION);
        int32_t h1 = C[0*src_plane_stride]+(((D[0*src_plane_stride]-C[0*src_plane_stride])*dx) >> SYN_LAYER_PRECISION);
        int32_t v = ((h1-h0)*dy) >> SYN_LAYER_PRECISION;
        dst[0*out_plane_stride] = ((h0+v)*gain) >> SYN_LAYER_PRECISION;

        h0 = A[1*src_plane_stride]+(((B[1*src_plane_stride]-A[1*src_plane_stride])*dx) >> SYN_LAYER_PRECISION);
        h1 = C[1*src_plane_stride]+(((D[1*src_plane_stride]-C[1*src_plane_stride])*dx) >> SYN_LAYER_PRECISION);
        v = ((h1-h0)*dy) >> SYN_LAYER_PRECISION;
        dst[1*out_plane_stride] = ((h0+v)*gain) >> SYN_LAYER_PRECISION;

        h0 = A[2*src_plane_stride]+(((B[2*src_plane_stride]-A[2*src_plane_stride])*dx) >> SYN_LAYER_PRECISION);
        h1 = C[2*src_plane_stride]+(((D[2*src_plane_stride]-C[2*src_plane_stride])*dx) >> SYN_LAYER_PRECISION);
        v = ((h1-h0)*dy) >> SYN_LAYER_PRECISION;
        dst[2*out_plane_stride] = ((h0+v)*gain) >> SYN_LAYER_PRECISION;

        if (add_residue)
        {
            dst[0*out_plane_stride] += raw[0*raw_plane_stride];
            dst[1*out_plane_stride] += raw[1*raw_plane_stride];
            dst[2*out_plane_stride] += raw[2*raw_plane_stride];
        }
    }

    const auto time_warp_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_warp = (time_warp_end-time_warp_start);
    time_warp_seconds += (float)elapsed_warp.count();
}

// final b-prediction
// add two pre-multiplied preds, multiply by alpha, add residue.
void bpred(struct frame_memory &bpred_result, struct frame_memory &raw_info, int raw_info_depth, struct frame_memory &pred0, struct frame_memory &pred1, int raw_gainidx)
{
    const auto time_bpred_start = std::chrono::steady_clock::now();
    int const h = raw_info.h;
    int const w = raw_info.w;
    int32_t *raw = raw_info.origin();
    int raw_stride = raw_info.stride;
    int raw_plane_stride = raw_info.plane_stride;

    bpred_result.update_to(h, w, 3, 0);
    int32_t *out = bpred_result.origin();

    int32_t *src0 = pred0.origin();
    int32_t *src1 = pred1.origin();
    int const src_plane_stride = pred0.plane_stride;

    if (pred0.plane_stride != pred1.plane_stride || pred0.stride != pred0.w)
    {
        printf("internal error\n");
        printf("p0 stride %d width %d; p1 %d %d\n", pred0.stride, pred0.w, pred1.stride, pred1.w);
        exit(1);
    }
    int32_t *dst = out; // first plane.
    int const raw_gain_offset = raw_gainidx * raw_plane_stride;

    for (int y = 0; y < h; y++, raw += raw_stride-w)
    for (int x = 0; x < w; x++, raw++, dst++, src0++, src1++)
    {
        int32_t gain = raw[raw_gain_offset]+(1<<(SYN_LAYER_PRECISION-1));
        if (gain < 0)
            gain = 0;
        else if (gain > (1<<SYN_LAYER_PRECISION))
            gain = (1<<SYN_LAYER_PRECISION);

        dst[0*src_plane_stride] = (((src0[0*src_plane_stride]+src1[0*src_plane_stride])*gain)>>SYN_LAYER_PRECISION) + raw[0*raw_plane_stride];
        dst[1*src_plane_stride] = (((src0[1*src_plane_stride]+src1[1*src_plane_stride])*gain)>>SYN_LAYER_PRECISION) + raw[1*raw_plane_stride];
        dst[2*src_plane_stride] = (((src0[2*src_plane_stride]+src1[2*src_plane_stride])*gain)>>SYN_LAYER_PRECISION) + raw[2*raw_plane_stride];
    }

    const auto time_bpred_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_bpred = (time_bpred_end-time_bpred_start);
    time_bpred_seconds += (float)elapsed_bpred.count();
}

// returns a yuv444 float, planar.
// raw_cc_output is either [3], [6] or [9] channel (intra, P, B)
// refx_444: float, planar.
void process_inter(struct frame_memory &pred, struct frame_memory &raw_cc_output, int im_h, int im_w, struct frame_memory *ref0_444, struct frame_memory *ref1_444, int flo_gain)
{
    if (ref1_444 == NULL)
    {
        // P
        // multiplies by alpha, adds residue.
        warp(pred, raw_cc_output, 6, *ref0_444, 3, 5, flo_gain, true);
    }
    else
    {
        struct frame_memory pred0;
        struct frame_memory pred1;

        // B
        // want to multiply by beta, do not add residue.
        warp(pred0, raw_cc_output, 9, *ref0_444, 3, 8, flo_gain, false);
        // want to multiply by 1-beta, do not add residue.
        warp(pred1, raw_cc_output, 9, *ref1_444, 6, -8, flo_gain, false);

        // want to add pred1, pred2, multiply by alpha, and add residue.
        bpred(pred, raw_cc_output, 9, pred0, pred1, 5);
    }
}

#ifdef CCDECAPI_CPU
int cc_decode_cpu(std::string &bitstream_filename, std::string &out_filename)
#else
#ifdef CCDECAPI_AVX2
int cc_decode_avx2(std::string &bitstream_filename, std::string &out_filename)
#else
#error must have one of CCDECAPI_CPU or CCDECAPI_AVX2 defined.
#endif
#endif
{
    if (bitstream_filename == "")
    {
        printf("must specify a bitstream\n");
        return 1;
    }

    const auto time_all_start = std::chrono::steady_clock::now();

    cc_bs bs;
    if (!bs.open(bitstream_filename))
    {
        printf("cannot open %s for reading\n", bitstream_filename.c_str());
        return 1;
    }

    struct cc_bs_gop_header &gop_header = bs.m_gop_header;
    struct frame_memory frames_444[gop_header.intra_period+1]; // we store all decoded frames for the moment.

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

    struct cc_frame_decoder frame_decoder(gop_header);
    for (int frame_coding_idx = 0; frame_coding_idx <= gop_header.intra_period; frame_coding_idx++)
    {
        // raw_cc_output either
        // [3] (I: residue)
        // [6] (P: residue+xy+alpha)
        // [9] (B: residue+xy+alpha+xy+beta)

        // from bitstream.
        cc_bs_frame *frame_symbols = bs.decode_frame();
        if (frame_symbols == NULL)
        {
            return 1;
        }

        // raw frame data.
        struct frame_memory *raw_cc_output = frame_decoder.decode_frame(frame_symbols);
        struct cc_bs_frame_header &frame_header = frame_symbols->m_frame_header;

        if (raw_cc_output == NULL)
        {
            printf("decoding failed\n");
            return 1;
        }

        struct frame_memory *frame_444p; // direct intra.
        struct frame_memory frame_444; // inter-created.

        // intra/inter processing.
        if (frame_coding_idx == 0)
        {
            // I
            frame_444p = raw_cc_output;
        }
        else
        {
            // search for references.
            // here, we distinguish P (syn: 6 outputs) and B (syn: 9 outputs)
            struct frame_memory *ref_prev = NULL;
            struct frame_memory *ref_next = NULL;
            // find prev for P and B.
            for (int idx = frame_header.display_index-1; idx >= 0; idx--)
                if (frames_444[idx].raw() != NULL)
                {
                    ref_prev = &frames_444[idx];
                    printf("refprev: %d\n", idx);
                    break;
                }
            // find next if B.
            if (frame_header.layers_synthesis[1].n_out_ft == 9)
                for (int idx = frame_header.display_index+1; idx <= gop_header.intra_period; idx++)
                    if (frames_444[idx].raw() != NULL)
                    {
                        ref_next = &frames_444[idx];
                        printf("refnext: %d\n", idx);
                        break;
                    }
            process_inter(frame_444, *raw_cc_output, gop_header.img_h, gop_header.img_w, ref_prev, ref_next, frame_header.flow_gain);
            frame_444p = &frame_444;
        }

        // YUV 420
        if (gop_header.frame_data_type == 1)
        {
            if (gop_header.bitdepth == 8)
            {
                unsigned char *frame_420 = convert_444_420_8b(*frame_444p);
                if (out_filename != "")
                    dump_yuv420_8b(frame_420, gop_header.img_h, gop_header.img_w, fout, frame_header.display_index);
                convert_420_444_8b(frames_444[frame_header.display_index], frame_420, gop_header.img_h, gop_header.img_w);
                delete[] frame_420;
            }
            else if (gop_header.bitdepth == 10)
            {
                unsigned short *frame_420 = convert_444_420_10b(*frame_444p);
                if (out_filename != "")
                    dump_yuv420_10b(frame_420, gop_header.img_h, gop_header.img_w, fout, frame_header.display_index);
                convert_420_444_10b(frames_444[frame_header.display_index], frame_420, gop_header.img_h, gop_header.img_w);
                delete[] frame_420;

            }
            else
            {
                printf("Unkown YUV bitdepth %d. Should be 8 or 10.", gop_header.bitdepth);
            }
        }
        // YUV 444
        else if (gop_header.frame_data_type == 2)
        {
            if (gop_header.bitdepth == 8)
            {
                unsigned char *raw_frame_444 = get_raw_444_8b(*frame_444p);
                if (out_filename != "")
                    dump_yuv444_8b(raw_frame_444, gop_header.img_h, gop_header.img_w, fout, frame_header.display_index);
                store_444_8b(frames_444[frame_header.display_index], raw_frame_444, gop_header.img_h, gop_header.img_w);
            }

            else if (gop_header.bitdepth == 10)
            {
                unsigned short *raw_frame_444 = get_raw_444_10b(*frame_444p);
                if (out_filename != "")
                    dump_yuv444_10b(raw_frame_444, gop_header.img_h, gop_header.img_w, fout, frame_header.display_index);
                store_444_10b(frames_444[frame_header.display_index], raw_frame_444, gop_header.img_h, gop_header.img_w);
            }
            else
            {
                printf("Unkown YUV bitdepth %d. Should be 8 or 10.", gop_header.bitdepth);
            }
        }

        else
        {
            // rgb
            if (out_filename != "")
                ppm_out(3, 3, *frame_444p, out_filename.c_str());
            if (gop_header.intra_period > 0)
            {
                printf("do not want to copy rgb in rgb video\n");
                exit(1);
            }
        }
        delete frame_symbols;

        const auto time_all_end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed_all = (time_all_end-time_all_start);
        time_all_seconds = (float)elapsed_all.count();

        printf("time: arm %g ups %g syn %g warp %g bpred %g all %g\n",
                time_arm_seconds, time_ups_seconds, time_syn_seconds, time_warp_seconds, time_bpred_seconds, time_all_seconds);
        fflush(stdout);
    }

    if (fout != NULL)
        fclose(fout);

    printf("decode done\n");

    return 0;
}


#if 0
char const *param_bitstream = "--input=";
char const *param_out       = "--output=";

void usage(char const *msg = NULL)
{
    if (msg != NULL)
        printf("%s\n", msg);
    printf("Usage:\n");
    printf("    %s: .hevc to decode\n", param_bitstream);
    printf("    %s: reconstruction if desired: ppm (image) or yuv420 (video) only\n", param_out);
    exit(msg != NULL);
}

int main(int argc, const char* argv[])
{

    std::string bitstream_filename;
    std::string out = "out";

    for (int i = 1; i < argc; i++)
    {
        if (strncmp(argv[i], param_bitstream, strlen(param_bitstream)) == 0)
            bitstream_filename = argv[i]+strlen(param_bitstream);
        else if (strncmp(argv[i], param_out, strlen(param_out)) == 0)
            out = argv[i]+strlen(param_out);
        else
        {
            std::string error_message = "unknown parameter ";
            error_message += argv[i];
            usage(error_message.c_str());
        }
    }

    if (bitstream_filename == "")
        usage("must specify a bitstream");

#ifdef CCDECAPI_CPU
    int result = cc_decode_cpu(bitstream_filename, out);
#else
    int result = cc_decode_avx2(bitstream_filename, out);
#endif
    printf("decode exit code: %d\n", result);
    return result;
}
#endif
