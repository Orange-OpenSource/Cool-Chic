

template <typename P>
void ups_refine_ks7_cpu(int ks, P *kw, frame_memory<P> &in, frame_memory<P> &out, int ups_src_precision, frame_memory<P> &tmp);
template <typename P>
void ups_refine_ksX_cpu(int ks, P *kw, frame_memory<P> &in, frame_memory<P> &out, int ups_src_precision, frame_memory<P> &tmp);

template <typename P>
void ups_upsample_ks8_cpu(int ksx2, P *kw, frame_memory<P> &in, frame_memory<P> &out, int out_plane, int ups_src_precision, frame_memory<P> &tmp);
template <typename P>
void ups_upsample_ksX_cpu(int ksx2, P *kw, frame_memory<P> &in, frame_memory<P> &out, int out_plane, int ups_src_precision, frame_memory<P> &tmp);
