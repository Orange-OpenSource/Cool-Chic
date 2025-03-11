
#include "frame-memory.h"

template <class P>
void frame_memory<P>::print_start(int plane, char const *msg, int n, int prec) const
{
    printf("x/%d (h%dx%d): %s\n", 1<<prec, n > 0 ? n : h, n > 0 ? n : w, msg == NULL ? "" : msg);
    if (std::is_same<P, float>::value)
    {
        for (int y = 0; y < (n > 0 ? n : h); y++)
        {
            for (int x = 0; x < (n > 0 ? n : w); x++)
                printf(" %g", (float)const_plane_origin(plane)[y*stride+x]);
            printf("\n");
        }
    }
    else
    {
        for (int y = 0; y < (n > 0 ? n : h); y++)
        {
            for (int x = 0; x < (n > 0 ? n : w); x++)
                printf(" %g", const_plane_origin(plane)[y*stride+x]/((1<<prec)+0.0));
            printf("\n");
        }
    }
    fflush(stdout);
}

template <typename P>
void frame_memory<P>::custom_pad_replicate_plane_in_place_i(int plane, int padlr, int padtb)
{
    P *scan_in = plane_origin(plane);
    P *scan_out = scan_in - padtb*stride-padlr;
    int w_in_padded = w+2*padlr;
    // leading lines
    for (int y = 0; y < padtb; y++)
    {
        if (y == 0)
        {
            // construct first padded line.
            for (int x = 0; x < padlr; x++)
            {
                *scan_out++ = *scan_in;
            }
            memcpy(scan_out, scan_in, w*sizeof(scan_out[0]));
            scan_out += w;
            for (int x = 0; x < padlr; x++)
            {
                *scan_out++ = scan_in[w-1];
            }
            scan_out += stride-w_in_padded;
        }
        else
        {
            // copy previous line.
            memcpy(scan_out, scan_out-stride, w_in_padded*sizeof(scan_out[0]));
            scan_out += stride;
        }
    }

    // internal lines: pad left and right edges.
    if (padlr > 0)
    {
        for (int y = 0; y < h; y++)
        {
            scan_out = scan_in-padlr;
            for (int x = 0; x < padlr; x++)
            {
                *scan_out++ = *scan_in;
            }
            scan_out += w;
            for (int x = 0; x < padlr; x++)
            {
                *scan_out++ = scan_in[w-1];
            }
            scan_in += stride;
            scan_out += stride-w_in_padded;
        }
    }
    else
    {
        scan_in += h*stride;
        scan_out += h*stride;
    }
    // trailing lines -- we copy the previous padded.
    scan_in -= stride; // beginning of last line, padding to the left.
    scan_out = scan_in+stride-padlr;
    for (int y = 0; y < padtb; y++)
    {
        memcpy(scan_out, scan_out-stride, w_in_padded*sizeof(scan_out[0]));
        scan_out += stride;
    }
}

template <typename P>
void frame_memory<P>::custom_pad_zero_plane_in_place_i(int plane, int padlr, int padtb)
{
    P *scan_out = plane_origin(plane) - padtb*stride-padlr;
    int w_in_padded = w+2*padlr;
    // leading lines
    for (int y = 0; y < padtb; y++)
    {
        memset(scan_out, 0, w_in_padded*sizeof(scan_out[0]));
        scan_out += stride;
    }

    // internal lines: pad left and right edges.
    if (padlr > 0)
    {
        for (int y = 0; y < h; y++)
        {
            memset(scan_out, 0, padlr*sizeof(scan_out[0]));
            memset(scan_out+padlr+w, 0, padlr*sizeof(scan_out[0]));
            scan_out += stride;
        }
    }
    else
    {
        scan_out += h*stride;
    }

    for (int y = 0; y < padtb; y++)
    {
        memset(scan_out, 0, w_in_padded*sizeof(scan_out[0]));
        scan_out += stride;
    }
}

template <typename P>
void frame_memory<P>::zero_pad(int plane, int pad)
{
    // !!! just the padding.
    memset(raw()+plane*plane_stride, 0, plane_stride*sizeof(P));
}

template <typename P>
void frame_memory<P>::zero_plane_content(int plane)
{
    // !!! just the content.
    memset(raw()+plane*plane_stride, 0, plane_stride*sizeof(P));
}


// for now, instantiate explit float and int32_t frames.
template
void frame_memory<int32_t>::print_start(int plane, char const *msg, int n, int prec) const;
template
void frame_memory<float>::print_start(int plane, char const *msg, int n, int prec) const;

template
void frame_memory<int32_t>::custom_pad_replicate_plane_in_place_i(int plane, int padlr, int padtb);
template
void frame_memory<float>::custom_pad_replicate_plane_in_place_i(int plane, int padlr, int padtb);

template
void frame_memory<int32_t>::custom_pad_zero_plane_in_place_i(int plane, int padlr, int padtb);
template
void frame_memory<float>::custom_pad_zero_plane_in_place_i(int plane, int padlr, int padtb);

template
void frame_memory<int32_t>::zero_pad(int plane, int pad);
template
void frame_memory<float>::zero_pad(int plane, int pad);

template
void frame_memory<int32_t>::zero_plane_content(int plane);
template
void frame_memory<float>::zero_plane_content(int plane);
