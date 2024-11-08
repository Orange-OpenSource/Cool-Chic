
#include "frame-memory.h"

void frame_memory::custom_pad_replicate_plane_in_place_i(int plane, int padlr, int padtb)
{
    int32_t *scan_in = plane_origin(plane);
    int32_t *scan_out = scan_in - padtb*stride-padlr;
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

void frame_memory::custom_pad_zero_plane_in_place_i(int plane, int padlr, int padtb)
{
    int32_t *scan_out = plane_origin(plane) - padtb*stride-padlr;
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

