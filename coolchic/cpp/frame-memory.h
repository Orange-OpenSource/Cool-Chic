
#include "common.h"

struct frame_memory
{
    buffer   raw_frame;
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
    int32_t *raw() { return raw_frame.data; }
    int32_t *origin() { return raw_frame.data+origin_idx; }
    int32_t *plane_origin(int plane = 0) { return origin()+plane*plane_stride; }
    int32_t *plane_pixel(int plane, int y, int x) { return plane_origin(plane)+y*stride+x; }
    int32_t *pad_origin(int plane, int pad) { return pad_origin(plane, pad, pad); }
    int32_t *pad_origin(int plane, int padlr, int padtb) { return raw_frame.data + origin_idx-padtb*stride-padlr + plane*plane_stride; }
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
    void print_start(int plane, char const *msg = NULL, int n = 5, int prec = 0)
    {
        printf("x/%d (h%dx%d): %s\n", 1<<prec, n > 0 ? n : h, n > 0 ? n : w, msg == NULL ? "" : msg);
        for (int y = 0; y < (n > 0 ? n : h); y++)
        {
            for (int x = 0; x < (n > 0 ? n : w); x++)
                printf(" %g", plane_origin(plane)[y*stride+x]/((1<<prec)+0.0));
            printf("\n");
        }
        fflush(stdout);
    }


    void custom_pad_replicate_plane_in_place_i(int plane, int pad) { custom_pad_replicate_plane_in_place_i(plane, pad, pad); }
    void custom_pad_zero_plane_in_place_i(int plane, int padlr, int padtb);
    void custom_pad_replicate_plane_in_place_i(int plane, int padlr, int padtb);
    void zero_pad(int plane, int pad);
    void zero_plane_content(int plane);
};

