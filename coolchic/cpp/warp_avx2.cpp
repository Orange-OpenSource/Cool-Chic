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
#include "warp_avx2.h"
#include "warp_cpu.h"

#include <immintrin.h>
#include "common_avx2.h"

template <typename P, int NTAPS>
inline void expand_coeffs(P coeffsyx[NTAPS][NTAPS], P coeffsx[NTAPS], P coeffsy[NTAPS])
{
    if constexpr(NTAPS != 8)
    {
        // !!! avx this.
        // We fill in the matrix from the coeffs.
        for (int y = 0; y < NTAPS; y++)
            for (int x = 0; x < NTAPS; x++)
                coeffsyx[y][x] = coeffsy[y]*coeffsx[x];
    }
    else
    {
        __m256 in_x = _mm256_loadu_ps(coeffsx);
        for (int y = 0; y < NTAPS; y++)
        {
            __m256 in_y = _mm256_set1_ps(coeffsy[y]);
            _mm256_storeu_ps(&coeffsyx[y][0], _mm256_mul_ps(in_y, in_x));
        }
    }
}

template <typename P, typename PACC, int NTAPS, bool SMALL_BLK>
void warp_ntap_avx2(struct frame_memory<P> &warp_result,
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

    if ((motion_blksz_shift < 3 && !SMALL_BLK) || (motion_blksz_shift >= 3 && SMALL_BLK))
    {
        printf("bad warp_ntap_avx2 call: motion_blksz_shift %d SMALL_BLK %d\n", motion_blksz_shift, SMALL_BLK);
    }
    if (SMALL_BLK)
        motion_blksz_shift = 2; // explicit.

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
    int const mot_w = frame_motion.w;

    P *dst = warp_result.origin(); // scans through first plane.
    int const blkstep = SMALL_BLK ? 2 : 1; // smallblk, 2 blocks at a time.

    for (int y = 0; y < h; y += (1<<motion_blksz_shift), motion += frame_motion.stride-mot_w, residual += (frame_residual.stride<<motion_blksz_shift)-w, dst += (warp_result.stride<<motion_blksz_shift)-w)
    for (int x = 0; x < w; x += blkstep*(1<<motion_blksz_shift), motion += blkstep, residual += blkstep*(1<<motion_blksz_shift), dst += blkstep*(1<<motion_blksz_shift))
    {
        int32_t basex;
        int32_t basey;
        int32_t tx; // qsteps, [0..1<<motion_q_shift)
        int32_t ty; // qsteps, [0..1<<motion_q_shift)

        //if (y >= 1 && x >= 2*4)
        //    exit(1);

        get_motion_info(motion[(raw_xyidx+0)*frame_motion.plane_stride]+x, motion_q_shift, basex, tx);
        get_motion_info(motion[(raw_xyidx+1)*frame_motion.plane_stride]+y, motion_q_shift, basey, ty);

        basex += global_flow[0];
        basey += global_flow[1];

        int32_t basex2;
        int32_t basey2;
        int32_t tx2; // qsteps, [0..1<<motion_q_shift)
        int32_t ty2; // qsteps, [0..1<<motion_q_shift)

        if constexpr(SMALL_BLK)
        {
            // next block's motion.
            get_motion_info(motion[(raw_xyidx+0)*frame_motion.plane_stride +1]+(x+(1<<motion_blksz_shift)), motion_q_shift, basex2, tx2);
            get_motion_info(motion[(raw_xyidx+1)*frame_motion.plane_stride +1]+y, motion_q_shift, basey2, ty2);

            basex2 += global_flow[0];
            basey2 += global_flow[1];
        }

        const auto time_warp_coeffget_start = std::chrono::steady_clock::now();
        // get x and y coeffs, same for all components.
        // can process a HxW sized block if motion is downsampled w.r.t. pixels.

        P *coeffsx = get_coeffs<P, NTAPS>(tx);
        P *coeffsy = get_coeffs<P, NTAPS>(ty);

        P coeffsyx[NTAPS][NTAPS]; // !!! currently for 8x8 filter or smaller.
        if constexpr(NTAPS <= 8)
        {
            expand_coeffs(coeffsyx, coeffsx, coeffsy);
        }

        P *coeffsx2;
        P *coeffsy2;
        P coeffsyx2[NTAPS][NTAPS]; // !!! currently for 8x8 filter or smaller.

        if constexpr(SMALL_BLK)
        {
            coeffsx2 = get_coeffs<P, NTAPS>(tx2);
            coeffsy2 = get_coeffs<P, NTAPS>(ty2);

            if constexpr(NTAPS <= 8)
            {
                expand_coeffs(coeffsyx2, coeffsx2, coeffsy2);
            }
        }

        const auto time_warp_coeffget_end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed_warp_coeffget = (time_warp_coeffget_end-time_warp_coeffget_start);
        time_warp_coeffget_seconds += (float)elapsed_warp_coeffget.count();

        // blocked motion comp -- do a block of pixels with the same motion.
        int res_suboffset = 0; // for indexing to y0x1, y1x0 and y1x1
        int dst_suboffset = 0;
        PACC gain = 1.0;
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
        PACC gain2 = 1.0;
        if (SMALL_BLK && !add_residue)
        {
            gain2 = motion[(raw_gainidx < 0 ? -raw_gainidx : raw_gainidx)*frame_motion.plane_stride +1]+0.5; // next block's gain.
            if (raw_gainidx < 0)
                gain2 = 1-gain2;
            if (gain2 < 0)
                gain2 = 0;
            else if (gain2 > 1.0)
                gain2 = 1.0;
        }

        // need to clip to image?  Look at block bounds.
        bool no_clip =
            get_limited(v_start_x, v_limit_x, basex+0-(NTAPS/2-1)) == basex+0-(NTAPS/2-1)
            && get_limited(v_start_x, v_limit_x, basex+((1<<motion_blksz_shift)-1)-(NTAPS/2-1)+NTAPS-1) == basex+((1<<motion_blksz_shift)-1)-(NTAPS/2-1)+NTAPS-1
            && get_limited(v_start_y, v_limit_y, basey+0-(NTAPS/2-1)) == basey+0-(NTAPS/2-1)
            && get_limited(v_start_y, v_limit_y, basey+((1<<motion_blksz_shift)-1)-(NTAPS/2-1)+NTAPS-1) == basey+((1<<motion_blksz_shift)-1)-(NTAPS/2-1)+NTAPS-1;
        if constexpr(SMALL_BLK)
        {
            // ensure next 2 blocks available.
            if (x+(1<<motion_blksz_shift) >= w)
                no_clip = false;
            // check motion in next block is within image.
            no_clip = no_clip
                && get_limited(v_start_x, v_limit_x, basex2+0-(NTAPS/2-1)) == basex2+0-(NTAPS/2-1)
                && get_limited(v_start_x, v_limit_x, basex2+((1<<motion_blksz_shift)-1)-(NTAPS/2-1)+NTAPS-1) == basex2+((1<<motion_blksz_shift)-1)-(NTAPS/2-1)+NTAPS-1
                && get_limited(v_start_y, v_limit_y, basey2+0-(NTAPS/2-1)) == basey2+0-(NTAPS/2-1)
                && get_limited(v_start_y, v_limit_y, basey2+((1<<motion_blksz_shift)-1)-(NTAPS/2-1)+NTAPS-1) == basey2+((1<<motion_blksz_shift)-1)-(NTAPS/2-1)+NTAPS-1;
        }
        if (no_clip)
        {
            // no clipping.
            P const *src_blk = &src[(basey-(NTAPS/2-1))*src_stride + basex-(NTAPS/2-1)];
            P const *src_blk2 = 0;
            if constexpr(SMALL_BLK)
                src_blk2 = &src[(basey2-(NTAPS/2-1))*src_stride + basex2-(NTAPS/2-1)];

            int const ystep = 1;
            int const xstep = 8;
            for (int suby = 0;
                 suby < (1<<motion_blksz_shift);
                 suby += ystep,
                    res_suboffset += frame_residual.stride*ystep-blkstep*(1<<motion_blksz_shift),
                    dst_suboffset += warp_result.stride*ystep-blkstep*(1<<motion_blksz_shift),
                    src_blk += src_stride*ystep-blkstep*(1<<motion_blksz_shift),
                    src_blk2 += src_stride*ystep-blkstep*(1<<motion_blksz_shift)) // back to x0 in subsequent line
            {
            for (int subx = 0; subx < blkstep*(1<<motion_blksz_shift); subx += xstep, res_suboffset += xstep, dst_suboffset += xstep, src_blk += xstep, src_blk2 += xstep)
            {
                P const *src_blk_c = src_blk;
                P const *src_blk2_c = src_blk2;
                __m256 gainpp;
                if (add_residue)
                {
                    // single per-pixel gain applied to all components.
                    gainpp = _mm256_loadu_ps(&residual[res_suboffset+raw_gainidx*frame_residual.plane_stride]);
                    gainpp = _mm256_add_ps(gainpp, _mm256_set1_ps(0.5));
                    // clamp.
                    gainpp = _mm256_blendv_ps(_mm256_setzero_ps(), gainpp, _mm256_cmp_ps(gainpp, _mm256_setzero_ps(), _CMP_GT_OS)); // <= 0 -> 0.
                    gainpp = _mm256_blendv_ps(_mm256_set1_ps(1.0), gainpp, _mm256_cmp_ps(gainpp, _mm256_set1_ps(1.0), _CMP_LT_OS)); // >= 1 -> 1.
                }

                for (int c = 0; c < 3; c++, src_blk_c += src_plane_stride, src_blk2_c += src_plane_stride)
                {
                    P const *src_c = src_blk_c;
                    P const *src2_c = src_blk2_c;
                    __m256 out8 = _mm256_setzero_ps();
                    if constexpr(SMALL_BLK)
                    {
                        for (int yy = 0; yy < NTAPS; yy++, src_c += src_stride, src2_c += src_stride)
                        for (int xx = 0; xx < NTAPS; xx++)
                        {
                            //__m256 inpix = _mm256_set_m128(_mm_loadu_ps(&src_c[xx]), _mm_loadu_ps(&src2_c[xx]));
                            __m256 inpix = _mm256_set_m128(_mm_loadu_ps(&src2_c[xx]), _mm_loadu_ps(&src_c[xx]));
                            __m256 coeff;
                            if constexpr(NTAPS <= 8)
                            {
                                // two different coeffs.
                                //coeff = _mm256_set_m128(_mm_set_ps1(coeffsyx[yy][xx]), _mm_set_ps1(coeffsyx2[yy][xx]));
                                coeff = _mm256_set_m128(_mm_set_ps1(coeffsyx2[yy][xx]), _mm_set_ps1(coeffsyx[yy][xx]));
                            }
                            else
                            {
                                // two different coeffs.
                                //coeff = _mm256_set_m128(_mm_set_ps1(coeffsy[yy]*coeffsx[xx]), _mm_set_ps1(coeffsy2[yy]*coeffsx2[xx]));
                                coeff = _mm256_set_m128(_mm_set_ps1(coeffsy2[yy]*coeffsx2[xx]), _mm_set_ps1(coeffsy[yy]*coeffsx[xx]));
                            }
                            out8 = _mm256_fmadd_ps(inpix, coeff, out8);
                        }
                    }
                    else
                    {
                        for (int yy = 0; yy < NTAPS; yy++, src_c += src_stride)
                        for (int xx = 0; xx < NTAPS; xx++)
                        {
                            __m256 inpix = _mm256_loadu_ps(&src_c[xx]);
                            __m256 coeff;
                            if constexpr(NTAPS <= 8)
                            {
                                coeff = _mm256_set1_ps(coeffsyx[yy][xx]);
                            }
                            else
                            {
                                coeff = _mm256_set1_ps(coeffsy[yy]*coeffsx[xx]);
                            }
                            out8 = _mm256_fmadd_ps(inpix, coeff, out8);
                        }
                    }

                    if (add_residue)
                    {
                        out8 = _mm256_mul_ps(out8, gainpp);

                        // final result, we clamp here.
                        __m256 respp;
                        respp = _mm256_loadu_ps(&residual[res_suboffset+c*frame_residual.plane_stride]);
                        out8 = _mm256_add_ps(out8, respp);
                        out8 = _mm256_blendv_ps(_mm256_setzero_ps(), out8, _mm256_cmp_ps(out8, _mm256_setzero_ps(), _CMP_GT_OS)); // <= 0 -> 0.
                        out8 = _mm256_blendv_ps(_mm256_set1_ps(1.0), out8, _mm256_cmp_ps(out8, _mm256_set1_ps(1.0), _CMP_LT_OS)); // >= 1 -> 1.
                    }
                    else
                    {
                        // single gain from motion.
                        if constexpr(SMALL_BLK)
                        {
                            // per-block motion
                            out8 = _mm256_mul_ps(out8, _mm256_set_m128(_mm_set_ps1(gain2), _mm_set_ps1(gain)));
                        }
                        else
                        {
                            out8 = _mm256_mul_ps(out8, _mm256_set1_ps(gain));
                        }
                    }

                    _mm256_storeu_ps(&dst[dst_suboffset+c*warp_result.plane_stride], out8);
                }
            } // subx
            } // suby
        }
        else
        {
            // clipping required during block processing.
            for (int suby = 0; suby < (1<<motion_blksz_shift); suby++, res_suboffset += frame_residual.stride-blkstep*(1<<motion_blksz_shift), dst_suboffset += warp_result.stride-blkstep*(1<<motion_blksz_shift)) // back to x0 in subsequent line
            for (int subx = 0; subx < blkstep*(1<<motion_blksz_shift); subx++, res_suboffset++, dst_suboffset++)
            {
                int limited_x[NTAPS];
                int limited_y[NTAPS];
                P gainpp;
                bool blk2 = false;
                P *cx = coeffsx;
                P *cy = coeffsy;

                if constexpr(SMALL_BLK)
                {
                    blk2 = subx >= (1<<motion_blksz_shift);
                    if (blk2)
                    {
                        cx = coeffsx2;
                        cy = coeffsy2;
                        // skip 2nd block content if outside image.
                        // the continue updates our scanning counters as if for a full block, and they will be correctly reset for the next line.
                        if (x+subx >= w)
                            continue;
                    }
                }

                // get vals clipped to image.
                for (int xy = 0; xy < NTAPS; xy++)
                {
                    if constexpr(SMALL_BLK)
                    {
                        if (blk2)
                        {
                            limited_x[xy] = get_limited(v_start_x, v_limit_x, basex2+subx-(1<<motion_blksz_shift)-(NTAPS/2-1)+xy);
                            limited_y[xy] = get_limited(v_start_y, v_limit_y, basey2+suby-(NTAPS/2-1)+xy)*src_stride;
                        }
                        else
                        {
                            limited_x[xy] = get_limited(v_start_x, v_limit_x, basex+subx-(NTAPS/2-1)+xy);
                            limited_y[xy] = get_limited(v_start_y, v_limit_y, basey+suby-(NTAPS/2-1)+xy)*src_stride;
                        }
                    }
                    else
                    {
                        limited_x[xy] = get_limited(v_start_x, v_limit_x, basex+subx-(NTAPS/2-1)+xy);
                        limited_y[xy] = get_limited(v_start_y, v_limit_y, basey+suby-(NTAPS/2-1)+xy)*src_stride;
                    }
                }

                if (add_residue)
                {
                    gainpp = residual[res_suboffset+raw_gainidx*frame_residual.plane_stride]+0.5;
                    if (gainpp < 0)
                        gainpp = 0;
                    else if (gainpp > 1.0)
                        gainpp = 1.0;
                }
                else
                {
                    if constexpr(SMALL_BLK)
                    {
                        gainpp = blk2 ? gain2 : gain;
                    }
                    else
                    {
                        gainpp = gain;
                    }
                }

                P const *src_c = src;
                for (int c = 0; c < 3; c++, src_c += src_plane_stride)
                {
                    P result = 0;
                    for (int yy = 0; yy < NTAPS; yy++)
                    {
                        P valsy_yy = 0;
                        for (int xx = 0; xx < NTAPS; xx++)
                        {
                            valsy_yy += src_c[limited_y[yy]+limited_x[xx]]*cx[xx];
                        }
                        result += valsy_yy*cy[yy];
                    }
                    result *= gainpp;
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
void warp_ntaps_avx2(struct frame_memory<P> &warp_result,
                     struct frame_memory<P> const &frame_residual, struct frame_memory<P> const &frame_motion, struct frame_memory<P> const &ref, int global_flow[2],
                     int raw_xyidx, int raw_gainidx, bool add_residue, int ntaps, int motion_blksz_shift, int motion_q)
{
    if (motion_blksz_shift < 2)
    {
        warp_ntaps_cpu<P, PACC>(warp_result, frame_residual, frame_motion, ref, global_flow, raw_xyidx, raw_gainidx, add_residue, ntaps, motion_blksz_shift, motion_q);
        return;
    }

    if (motion_blksz_shift == 2)
    {
        switch(ntaps)
        {
        case 2:
            warp_ntap_avx2<P, PACC, 2, true>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 4:
            warp_ntap_avx2<P, PACC, 4, true>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 6:
            warp_ntap_avx2<P, PACC, 6, true>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 8:
            warp_ntap_avx2<P, PACC, 8, true>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 10:
            warp_ntap_avx2<P, PACC, 10, true>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 12:
            warp_ntap_avx2<P, PACC, 12, true>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 14:
            warp_ntap_avx2<P, PACC, 14, true>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 16:
            warp_ntap_avx2<P, PACC, 16, true>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        default:
            printf("unsupported interpolation #taps: received ntaps=%d\n", ntaps);
            exit(1);
        }
    }
    else
    {
        switch (ntaps)
        {
        case 2:
            warp_ntap_avx2<P, PACC, 2, false>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 4:
            warp_ntap_avx2<P, PACC, 4, false>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 6:
            warp_ntap_avx2<P, PACC, 6, false>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 8:
            warp_ntap_avx2<P, PACC, 8, false>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 10:
            warp_ntap_avx2<P, PACC, 10, false>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 12:
            warp_ntap_avx2<P, PACC, 12, false>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 14:
            warp_ntap_avx2<P, PACC, 14, false>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        case 16:
            warp_ntap_avx2<P, PACC, 16, false>(warp_result, frame_residual, frame_motion, ref, global_flow, motion_blksz_shift, motion_q, raw_xyidx, raw_gainidx, add_residue);
            break;
        default:
            printf("unsupported interpolation #taps: received ntaps=%d\n", ntaps);
            exit(1);
        }
    }
}

// instantiate float float.
template
void warp_ntaps_avx2<float, float>(
    struct frame_memory<float> &warp_result,
    struct frame_memory<float> const &frame_residual, struct frame_memory<float> const &frame_motion, struct frame_memory<float> const &ref, int global_flow[2],
    int raw_xyidx, int raw_gainidx, bool add_residue, int ntaps, int motion_blksz_shift, int motion_q);
