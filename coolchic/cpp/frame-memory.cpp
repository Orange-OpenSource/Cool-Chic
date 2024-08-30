
#include "frame-memory.h"

void frame_memory::custom_pad_replicate_plane_in_place_i(int plane, int pad)
{
    int32_t *scan_in = plane_origin(plane);
    int32_t *scan_out = scan_in - pad*stride-pad;
    int w_in_padded = w+2*pad;
    // leading lines
    for (int y = 0; y < pad; y++)
    {
        if (y == 0)
        {
            // construct first padded line.
            for (int x = 0; x < pad; x++)
            {
                *scan_out++ = *scan_in;
            }
            memcpy(scan_out, scan_in, w*sizeof(scan_out[0]));
            scan_out += w;
            for (int x = 0; x < pad; x++)
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
    for (int y = 0; y < h; y++)
    {
        scan_out = scan_in-pad;
        for (int x = 0; x < pad; x++)
        {
            *scan_out++ = *scan_in;
        }
        scan_out += w;
        for (int x = 0; x < pad; x++)
        {
            *scan_out++ = scan_in[w-1];
        }
        scan_in += stride;
    }
    // trailing lines -- we copy the previous padded.
    scan_in -= stride; // beginning of last line, padding to the left.
    scan_out = scan_in+stride-pad;
    for (int y = 0; y < pad; y++)
    {
        memcpy(scan_out, scan_out-stride, w_in_padded*sizeof(scan_out[0]));
        scan_out += stride;
    }
}

void frame_memory::zero_pad(int plane, int pad)
{
    // !!! just the padding.
    memset(raw()+plane*plane_stride, 0, plane_stride*sizeof(int32_t));
}

void frame_memory::zero_plane_content(int plane)
{
    // !!! just the content.
    memset(raw()+plane*plane_stride, 0, plane_stride*sizeof(int32_t));
}

