

void ups_refine_ks7_cpu(int ks, int32_t *kw, frame_memory &in, frame_memory &out, int ups_src_precision, frame_memory &tmp);
void ups_refine_ksX_cpu(int ks, int32_t *kw, frame_memory &in, frame_memory &out, int ups_src_precision, frame_memory &tmp);

void ups_upsample_ks8_cpu(int ksx2, int32_t *kw, frame_memory &in, frame_memory &out, int out_plane, int ups_src_precision, frame_memory &tmp);
void ups_upsample_ksX_cpu(int ksx2, int32_t *kw, frame_memory &in, frame_memory &out, int out_plane, int ups_src_precision, frame_memory &tmp);
