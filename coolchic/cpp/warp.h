
template <typename P, typename PACC>
void warp_ntaps(struct frame_memory<P> &warp_result,
                struct frame_memory<P> const &frame_residual, struct frame_memory<P> const &frame_motion, struct frame_memory<P> const &ref, int global_flow[2],
                int raw_xyidx, int raw_gainidx, bool add_residue, int ntaps, int motion_blksz_shift, int motion_q);
