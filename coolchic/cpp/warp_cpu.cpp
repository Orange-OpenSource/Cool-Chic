/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#include <stdlib.h>
#include <chrono> // timing.
#include <math.h>

#include "common.h"
#include "warp_common.h"
#include "warp_cpu.h"

// ntap sampling.
// ref: float, planar.
// raw_xyidx: where to get xy from motion
// raw_gain_idx: where to get alpha in residual if add_residue (P processing)
// raw_gain_idx: where to get beta (+ve) or 1-beta (-ve) in motion if !add_residue (B processing)
//
// note: the motion can be downsampled w.r.t. the pred and residal.
template <typename P, typename PACC, int NTAPS>
void warp_ntap_cpu(struct frame_memory<P> &warp_result,
                   struct frame_memory<P> const &frame_residual, struct frame_memory<P> const &frame_motion, struct frame_memory<P> const &ref,
                   int global_flow[2], int motion_blksz_shift, int motion_q_shift,
                   int raw_xyidx, int raw_gainidx, bool add_residue)
{
    const auto time_warp_start = std::chrono::steady_clock::now();

    warp_result.update_to(ref.h, ref.w, 0, 3);

    const auto time_warp_coeffgen_start = std::chrono::steady_clock::now();
    generate_coeffs<P, NTAPS>(motion_q_shift); // all motion-quantized values.
    const auto time_warp_coeffgen_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_warp_coeffgen = (time_warp_coeffgen_end-time_warp_coeffgen_start);
    time_warp_coeffgen_seconds += (float)elapsed_warp_coeffgen.count();

    P const *src = ref.const_origin();
    // We reduce the dimensions of the reference with global_flow. Accessing outside will pad correctly.
    // a couple of virtual tests.
    int const v_start_x = global_flow[0] < 0 ? 0 : (global_flow[0] < ref.w ? global_flow[0] : ref.w-1);
    int const v_limit_x = global_flow[0] < 0 ? (ref.w+global_flow[0] > 0 ? ref.w+global_flow[0] : 1) : ref.w;
    int const v_start_y = global_flow[1] < 0 ? 0 : (global_flow[1] < ref.h ? global_flow[1] : ref.h-1);
    int const v_limit_y = global_flow[1] < 0 ? (ref.h+global_flow[1] > 0 ? ref.h+global_flow[1] : 1) : ref.h;

    int const src_stride = ref.stride; // ref.stride;
    int const src_plane_stride = ref.plane_stride; // ref.plane_stride;

    P const *residual = frame_residual.const_origin();
    P const *motion = frame_motion.const_origin();
    int const h = frame_residual.h;
    int const w = frame_residual.w;

    // downsampled motion?  (normal case)
    int const mot_w = frame_motion.w;

    P *dst = warp_result.origin(); // scans through first plane.

    for (int y = 0; y < h; y += (1<<motion_blksz_shift), motion += frame_motion.stride-mot_w, residual += (frame_residual.stride<<motion_blksz_shift)-w, dst += (warp_result.stride<<motion_blksz_shift)-w)
    for (int x = 0; x < w; x += (1<<motion_blksz_shift), motion++, residual += (1<<motion_blksz_shift), dst += (1<<motion_blksz_shift))
    {
        int32_t basex;
        int32_t basey;
        int32_t tx; // qsteps, [0..1<<motion_q_shift)
        int32_t ty; // qsteps, [0..1<<motion_q_shift)

        if constexpr(!std::is_same<P, float>::value)
        {
            printf("Only float warp supported for the moment.\n");
            exit(1);
#if 0
            // want a pixel fractional position.
            px = buffer<P>::integerize15(motion[(raw_xyidx+0)*frame_motion.plane_stride], scale_motion) + (x<<15);
            py = buffer<P>::integerize15(motion[(raw_xyidx+1)*frame_motion.plane_stride], scale_motion) + (y<<15);
            basex0 = px < 0 ? ((px-((1<<15)-1)) >> 15) : (px >> 15);
            dx = px-(basex0<<15);

            basey0 = py < 0 ? ((py-((1<<15)-1)) >> 15) : (py >> 15);
            dy = py-(basey0<<15);
#endif
        }
        else
        {
            get_motion_info(motion[(raw_xyidx+0)*frame_motion.plane_stride]+x, motion_q_shift, basex, tx);
            get_motion_info(motion[(raw_xyidx+1)*frame_motion.plane_stride]+y, motion_q_shift, basey, ty);
        }

        basex += global_flow[0];
        basey += global_flow[1];

        if constexpr(!std::is_same<P, float>::value)
        {
            printf("Only float warp supported for the moment.\n");
            exit(1);
#if 0
            if (add_residue)
            {
                gain = buffer<P>::integerize15(residual[raw_gainidx*frame_residual.plane_stride], scale_residual)+(1<<(15-1));
            }
            else
            {
                gain = buffer<P>::integerize15(motion[(raw_gainidx < 0 ? -raw_gainidx : raw_gainidx)*frame_motion.plane_stride], scale_motion)+(1<<(15-1));
                if (raw_gainidx < 0)
                    gain = (1<<15)-gain;
            }
            if (gain < 0)
                gain = 0;
            else if (gain > (1<<15))
                gain = (1<<15);
#endif
        }
        else
        {
#if 0 // migrated to blocked motion comp
            if (add_residue)
            {
                gain = residual[raw_gainidx*frame_residual.plane_stride]+0.5;
            }
            else
            {
                gain = motion[(raw_gainidx < 0 ? -raw_gainidx : raw_gainidx)*frame_motion.plane_stride]+0.5;
                if (raw_gainidx < 0)
                    gain = 1-gain;
            }
            if (gain < 0)
                gain = 0;
            else if (gain > 1.0)
                gain = 1.0;
#endif
        }

        const auto time_warp_coeffget_start = std::chrono::steady_clock::now();
        // get x and y coeffs, same for all components.
        // can process a HxW sized block if motion is downsampled w.r.t. pixels.

        P *coeffsx = get_coeffs<P, NTAPS>(tx);
        P *coeffsy = get_coeffs<P, NTAPS>(ty);

        const auto time_warp_coeffget_end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed_warp_coeffget = (time_warp_coeffget_end-time_warp_coeffget_start);
        time_warp_coeffget_seconds += (float)elapsed_warp_coeffget.count();

        // blocked motion comp -- do a block of pixels with the same motion.
        int res_suboffset = 0; // for indexing to y0x1, y1x0 and y1x1
        int dst_suboffset = 0;
        PACC gain = 1;
        if (!add_residue)
        {
            gain = motion[(raw_gainidx < 0 ? -raw_gainidx : raw_gainidx)*frame_motion.plane_stride]+0.5;
            if (raw_gainidx < 0)
                gain = 1-gain;
            if (gain < 0)
                gain = 0;
            else if (gain > 1.0)
                gain = 1.0;
        }

        // need to clip to image?  Look at block bounds.
        if (get_limited(v_start_x, v_limit_x, basex+0-(NTAPS/2-1)) == basex+0-(NTAPS/2-1)
            && get_limited(v_start_x, v_limit_x, basex+((1<<motion_blksz_shift)-1)-(NTAPS/2-1)+NTAPS-1) == basex+((1<<motion_blksz_shift)-1)-(NTAPS/2-1)+NTAPS-1
            && get_limited(v_start_y, v_limit_y, basey+0-(NTAPS/2-1)) == basey+0-(NTAPS/2-1)
            && get_limited(v_start_y, v_limit_y, basey+((1<<motion_blksz_shift)-1)-(NTAPS/2-1)+NTAPS-1) == basey+((1<<motion_blksz_shift)-1)-(NTAPS/2-1)+NTAPS-1)
        {
            // no clipping.
            P const *src_blk = &src[(basey-(NTAPS/2-1))*src_stride + basex-(NTAPS/2-1)];
            for (int suby = 0; suby < (1<<motion_blksz_shift); suby++, res_suboffset += frame_residual.stride-(1<<motion_blksz_shift), dst_suboffset += warp_result.stride-(1<<motion_blksz_shift), src_blk += src_stride-(1<<motion_blksz_shift)) // back to x0 in subsequent line
            {
            for (int subx = 0; subx < (1<<motion_blksz_shift); subx++, res_suboffset++, dst_suboffset++, src_blk++)
            {
                if (add_residue)
                {
                    gain = residual[res_suboffset+raw_gainidx*frame_residual.plane_stride]+0.5;
                    if (gain < 0)
                        gain = 0;
                    else if (gain > 1.0)
                        gain = 1.0;
                }

                P const *src_blk_c = src_blk;
                for (int c = 0; c < 3; c++, src_blk_c += src_plane_stride)
                {
                    P const *src_c = src_blk_c;
                    P result = 0;
                    src_c = src_blk_c;
                    for (int yy = 0; yy < NTAPS; yy++, src_c += src_stride)
                    {
                        P valsy_yy = 0;
                        for (int xx = 0; xx < NTAPS; xx++)
                            valsy_yy += src_c[xx]*coeffsx[xx];
                        result += valsy_yy*coeffsy[yy];
                    }
                    result *= gain;
                    if (add_residue)
                    {
                        // final result, we clamp here.
                        result += residual[res_suboffset+c*frame_residual.plane_stride];
                        if (result < 0)
                            result = 0;
                        else if (result > 1)
                            result = 1;
                    }
                    dst[dst_suboffset+c*warp_result.plane_stride] = result;
                }
            } // subx
            } // suby
        }
        else
        {
            // clipping required during block processing.
            for (int suby = 0; suby < (1<<motion_blksz_shift); suby++, res_suboffset += frame_residual.stride-(1<<motion_blksz_shift), dst_suboffset += warp_result.stride-(1<<motion_blksz_shift)) // back to x0 in subsequent line
            for (int subx = 0; subx < (1<<motion_blksz_shift); subx++, res_suboffset++, dst_suboffset++)
            {
                int limited_x[NTAPS];
                int limited_y[NTAPS];

                // get vals clipped to image.
                for (int xy = 0; xy < NTAPS; xy++)
                {
                    limited_x[xy] = get_limited(v_start_x, v_limit_x, basex+subx-(NTAPS/2-1)+xy);
                    limited_y[xy] = get_limited(v_start_y, v_limit_y, basey+suby-(NTAPS/2-1)+xy)*src_stride;
                }

                if (add_residue)
                {
                    gain = residual[res_suboffset+raw_gainidx*frame_residual.plane_stride]+0.5;
                    if (gain < 0)
                        gain = 0;
                    else if (gain > 1.0)
                        gain = 1.0;
                }

                P const *src_c = src;
                for (int c = 0; c < 3; c++, src_c += src_plane_stride)
                {
                    P result = 0;
                    for (int yy = 0; yy < NTAPS; yy++)
                    {
                        P valsy_yy = 0;
                        for (int xx = 0; xx < NTAPS; xx++)
                            valsy_yy += src_c[limited_y[yy]+limited_x[xx]]*coeffsx[xx];
                        result += valsy_yy*coeffsy[yy];
                    }
                    result *= gain;
                    if (add_residue)
                    {
                        // final result, we clamp here.
                        result += residual[res_suboffset+c*frame_residual.plane_stride];
                        if (result < 0)
                            result = 0;
                        else if (result > 1)
                            result = 1;
                    }

                    dst[dst_suboffset+c*warp_result.plane_stride] = result;
                }
            }
        }
    }

    const auto time_warp_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_warp = (time_warp_end-time_warp_start);
    time_warp_seconds += (float)elapsed_warp.count();
}

template <typename P, typename PACC>
void warp_ntaps_cpu(struct frame_memory<P> &warp_result,
                    struct frame_memory<P> const &frame_residual, struct frame_memory<P> const &frame_motion, struct frame_memory<P> const &ref, int global_flow[2],
                    int raw_xyidx, int raw_gainidx, bool add_residue, int ntaps, int motion_blksz_shift, int motion_q)
{
    switch (ntaps)
    {
    case 2:
        warp_ntap_cpu<P, PACC, 2>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
        break;
    case 4:
        warp_ntap_cpu<P, PACC, 4>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
        break;
    case 6:
        warp_ntap_cpu<P, PACC, 6>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
        break;
    case 8:
        warp_ntap_cpu<P, PACC, 8>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
        break;
    case 10:
        warp_ntap_cpu<P, PACC, 10>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
        break;
    case 12:
        warp_ntap_cpu<P, PACC, 12>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
        break;
    case 14:
        warp_ntap_cpu<P, PACC, 14>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
        break;
    case 16:
        warp_ntap_cpu<P, PACC, 16>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
        break;
    default:
        printf("unsupported interpolation #taps: received ntaps=%d\n", ntaps);
        exit(1);
    }
}

// instantiate float float.
template
void warp_ntaps_cpu<float, float>(
    struct frame_memory<float> &warp_result,
    struct frame_memory<float> const &frame_residual, struct frame_memory<float> const &frame_motion, struct frame_memory<float> const &ref, int global_flow[2],
    int raw_xyidx, int raw_gainidx, bool add_residue, int ntaps, int motion_blksz_shift, int motion_q);
