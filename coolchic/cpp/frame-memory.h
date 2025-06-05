
#include "common.h"

// P is expected to be int32_t or float
// underlying buffer data is for the moment just int32_t.
template <typename P>
struct frame_memory
{
    buffer<P>raw_frame;
    int      origin_idx;
    int      h;
    int      w;
    int      pad;
    int      planes;
    int      stride;
    int      plane_stride;
public:
    frame_memory():
        origin_idx(0),
        h(0),
        w(0),
        pad(0),
        planes(0),
        stride(0),
        plane_stride(0)
        {};
    ~frame_memory() {};
public:
    void update_to(int frame_h, int frame_w, int frame_pad, int frame_planes)
    {
        h = frame_h;
        w = frame_w;
        pad = frame_pad;
        planes = frame_planes;
        stride = frame_pad+frame_w+frame_pad;
        plane_stride = (frame_pad+frame_h+frame_pad)*stride;
        origin_idx = frame_pad*stride+frame_pad;
        raw_frame.update_to(frame_planes*plane_stride);
    }
    void update_to(struct frame_memory &src) { update_to(src.h, src.w, src.pad, src.planes); }
    void copy_from(struct frame_memory const &src, int nplanes) { update_to(src.h, src.w, src.pad, nplanes); memcpy(raw_frame.data, src.raw_frame.data, nplanes*plane_stride*sizeof(raw_frame.data[0])); }
    void copy_from(struct frame_memory const &src) { copy_from(src, src.planes); }
    P *raw() { return raw_frame.data; }
    P const *const_raw() const { return (P const *)raw_frame.data; }
    P const *const_origin() const { return const_raw()+origin_idx; }
    P *origin() { return raw()+origin_idx; }
    P *plane_origin(int plane = 0) { return origin()+plane*plane_stride; }
    P const *const_plane_origin(int plane = 0) const { return const_origin()+plane*plane_stride; }
    P *plane_pixel(int plane, int y, int x) { return plane_origin(plane)+y*stride+x; }
    P const *const_plane_pixel(int plane, int y, int x) const { return const_plane_origin(plane)+y*stride+x; }
    P *pad_origin(int plane, int pad) { return pad_origin(plane, pad, pad); }
    P *pad_origin(int plane, int padlr, int padtb) { return raw() + origin_idx-padtb*stride-padlr + plane*plane_stride; }
#if 0
    void print_ranges(char const *msg, int nplanes, int precision)
        {
            int minval = plane_origin(0)[0];
            int maxval = plane_origin(0)[0];
            for (int p = 0; p < nplanes; p++)
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                if (plane_origin(p)[y*stride+x] < minval)
                    minval = plane_origin(p)[y*stride+x];
                else if (plane_origin(p)[y*stride+x] > maxval)
                    maxval = plane_origin(p)[y*stride+x];
            }
            printf("%s: minval=%g maxval=%g\n", msg == NULL ? "" : msg, (float)minval/(1<<precision), (float)maxval/(1<<precision));
        }
#endif
    void print_start(int plane, char const *msg = NULL, int n = 5, int prec = 0) const;

    void custom_pad_replicate_plane_in_place_i(int plane, int pad) { custom_pad_replicate_plane_in_place_i(plane, pad, pad); }
    void custom_pad_zero_plane_in_place_i(int plane, int padlr, int padtb);
    void custom_pad_replicate_plane_in_place_i(int plane, int padlr, int padtb);
    void zero_pad(int plane, int pad);
    void zero_plane_content(int plane);
};
