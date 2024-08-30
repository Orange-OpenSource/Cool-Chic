/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <pybind11/stl.h>
#include <string>
#include <memory.h>

#include "TDecBinCoderCABAC.h" // decoding weights/biases
#include "TEncBinCoderCABAC.h" // encoding weights/bases, latents. 
#include "Contexts.h"
#include "common.h" // needed now for weights/biases decode.
#include "cc-contexts.h"
#include "BitStream.h"

// encode weights and biases to a file.
// use count >= 0 will use that count, otherwise a search will be performed to reduce rate.
// returns the count used.
int cc_code_wb_bac(std::string &out_file, std::vector<int> &x, int use_count);

// encode latents layer to a file.
void cc_code_latent_layer_bac(
    std::string &out_file,
    std::vector<int> &x,
    std::vector<int> &mu,
    std::vector<int>  &log_scale,
    int layer_height, int layer_width,
    int hls_sig_blksize);

// create a decode object for weights or biases from a file.
class cc_decode_wb {
public:
    cc_decode_wb(const std::string &in_file);
    virtual ~cc_decode_wb() {}
public:
    std::vector<int> decode_wb_continue(int n_weights, int scale_index);

private:
    InputBitstream m_bs;
    TDecBinCABAC m_cabac;
};

PYBIND11_MODULE(ccencapi, m) {
    m.doc() = "ccencoding"; // optional module docstring
    m.def("cc_code_wb_bac", &cc_code_wb_bac, "code weights and biases");
    m.def("cc_code_latent_layer_bac", &cc_code_latent_layer_bac, "code a latent layer");

    py::class_<cc_decode_wb>(m, "cc_decode_wb")
        .def(py::init<const std::string &>())
        .def("decode_wb_continue", &cc_decode_wb::decode_wb_continue);
}

void code_val(TEncBinCABAC &layer_BAC, MuSigGTs *coding_ctxs, int val_to_code)
{
    int abs_val_to_code = val_to_code < 0 ? -val_to_code : val_to_code;
    if (abs_val_to_code == 0)
        layer_BAC.encodeBin(coding_ctxs->m_gt0, 0);
    else
    {
        // significant.
        layer_BAC.encodeBin(coding_ctxs->m_gt0, 1);
        if (abs_val_to_code <= 1)
            layer_BAC.encodeBin(coding_ctxs->m_gt1, 0);
        else
        {
            layer_BAC.encodeBin(coding_ctxs->m_gt1, 1);
            if (abs_val_to_code <= 2)
                layer_BAC.encodeBin(coding_ctxs->m_gt2, 0);
            else
            {
                layer_BAC.encodeBin(coding_ctxs->m_gt2, 1);
                if (abs_val_to_code <= 3)
                    layer_BAC.encodeBin(coding_ctxs->m_gt3, 0);
                else
                {
                    layer_BAC.encodeBin(coding_ctxs->m_gt3, 1);
                    layer_BAC.encodeExGolomb(abs_val_to_code-3-1, 0);
                }
            }
        }
        // sign when significant.
        layer_BAC.encodeBin(coding_ctxs->m_ppos, val_to_code < 0 ? 1 : 0);
    }
}

// weights and biases coding -- always a mu of zero.
// we return the best index.
int cc_code_wb_bac(std::string &out_name, std::vector<int> &xs, int use_count)
{
    // !!! check for all zero, emit empty file.
    bool all_zero = true;
    for (int i = 0; i < (int)xs.size(); i++)
    {
        if (xs[i] != 0)
        {
            all_zero = false;
            break;
        }
    }
    if (all_zero)
    {
        printf("all weights/biases zero -- think of empty file\n");
    }

    TEncBinCABAC layer_BAC;

    // Just use exgolomb with different counts
    int best_exgolomb_count = -1;
    std::vector<unsigned char> best_exgolomb_bytes;
    int test_min = 0;
    int test_max = 12;
    if (use_count >= 0)
        test_min = test_max = use_count;
    for (int exgolomb_count = test_min; exgolomb_count <= test_max; exgolomb_count++)
    {
        //auto layer_BAC =  CABACEncoder();
        OutputBitstream bsBAC;
        layer_BAC.init(&bsBAC);
        layer_BAC.start();

        for (int i = 0; i < (int)xs.size(); i++)
        {
            layer_BAC.encodeExGolomb(abs(xs[i]), exgolomb_count);
            if (xs[i] != 0)
                layer_BAC.encodeBinEP(xs[i] < 0 ? 1 : 0);
        }
        layer_BAC.encodeBinTrm(1);
        layer_BAC.finish();
        bsBAC.write(1, 1);
        bsBAC.writeAlignZero();
        if (best_exgolomb_count < 0 || bsBAC.getFifo().size() < best_exgolomb_bytes.size())
        {
            best_exgolomb_count = exgolomb_count;
            best_exgolomb_bytes = bsBAC.getFifo();
            printf("better exgolomb bytes %d at count=%d\n", (int)best_exgolomb_bytes.size(), best_exgolomb_count);
        }
    }

    // code.
    FILE *fout = fopen(out_name.c_str(), "wb");
    if (fout == NULL)
    {
        printf("Cannot open %s for writing\n", out_name.c_str());
        exit(1);
    }
    if (fwrite(best_exgolomb_bytes.data(), best_exgolomb_bytes.size(), 1, fout) != 1)
    {
        printf("Write failure to %s\n", out_name.c_str());
        exit(1);
    }
    fclose(fout);
    printf("%s created\n", out_name.c_str());

    // return best sig index used for coding
    return best_exgolomb_count;
}

void cc_code_latent_layer_bac(
    std::string &out_name,
    std::vector<int> &xs,
    std::vector<int> &mus,
    std::vector<int> &log_scales,
    int layer_height, int layer_width,
    int hls_sig_blksize)
{
    printf("called cc_code_latent_layer_bac: file=%s\n", out_name.c_str());

    // get significant blocks.
    bool hls_sig_update = hls_sig_blksize < 0;
    if (hls_sig_update)
        hls_sig_blksize = -hls_sig_blksize;

    int hls_sig_shift = 0;
    while ((1<<hls_sig_shift) < hls_sig_blksize)
        hls_sig_shift++;

    int nby = 1;
    int nbx = 1;
    if (hls_sig_blksize != 0)
    {
        nby = (layer_height+hls_sig_blksize-1)/hls_sig_blksize;
        nbx = (layer_width+hls_sig_blksize-1)/hls_sig_blksize;
    }
    int *blk_sig = new int[nby*nbx];
    int *blk_flat = new int[nby*nbx];
    memset(blk_sig, 1, nby*nbx*sizeof(blk_sig[0]));
    memset(blk_flat, 0, nby*nbx*sizeof(blk_flat[0]));

    OutputBitstream bsBAC;
    TEncBinCABAC layer_BAC;
    layer_BAC.init(&bsBAC);
    layer_BAC.start();

#if ENTROPY_CODING_DEBUG
    g_epbits = 0;
    g_trmbits = 0;
    for (int m = 0; m < N_MUQ+1; m++)
    {
        for (int s = 0; s < N_SIGQ; s++)
        {
            g_contexts[m][s].m_gt0.m_binCnt = 0; g_contexts[m][s].m_gt0.m_zeroCnt = 0; g_contexts[m][s].m_gt0.m_numBits = 0;
            g_contexts[m][s].m_gt1.m_binCnt = 0; g_contexts[m][s].m_gt1.m_zeroCnt = 0; g_contexts[m][s].m_gt1.m_numBits = 0;
            g_contexts[m][s].m_gt2.m_binCnt = 0; g_contexts[m][s].m_gt2.m_zeroCnt = 0; g_contexts[m][s].m_gt2.m_numBits = 0;
            g_contexts[m][s].m_gt3.m_binCnt = 0; g_contexts[m][s].m_gt3.m_zeroCnt = 0; g_contexts[m][s].m_gt3.m_numBits = 0;
            g_contexts[m][s].m_ppos.m_binCnt = 0; g_contexts[m][s].m_ppos.m_zeroCnt = 0; g_contexts[m][s].m_ppos.m_numBits = 0;
        }
    }
#endif

    if (nby != 1 || nbx != 1)
    {
        // sigs and flats.
        int n_zero = 0;
        int n_flat = 0;
        for (int by = 0; by < nby; by++)
        {
            for (int bx = 0; bx < nbx; bx++)
            {
                bool sig = false;
                bool flat = true;
                int first_val = xs[by*hls_sig_blksize*layer_width + bx*hls_sig_blksize];
                for (int y = by*hls_sig_blksize; y < (by+1)*hls_sig_blksize && y < layer_height; y++)
                    for (int x = bx*hls_sig_blksize; x < (bx+1)*hls_sig_blksize && x < layer_width; x++)
                    {
                        sig = sig || xs[y*layer_width+x] != 0;
                        flat = flat && xs[y*layer_width+x] == first_val;
                    }
                blk_sig[by*nbx+bx] = sig;
                blk_flat[by*nbx+bx] = flat;
                if (!sig)
                    n_zero++;
                else if (flat)
                    n_flat++;
            }
        }

        // want to bother?  For significant blocks, we now say no.
        // SIG
        printf("nz %d vs %d\n", n_zero, nby*nbx);
        //if (n_zero <= nby*nbx/20)
        if (1) // no longer use significance blocks.
        {
            //not-enough zero-blocks, don't signal significance, assume everything is significant.
            layer_BAC.encodeBinEP(0);
            memset(blk_sig, 1, nby*nbx*sizeof(blk_sig[0])); // non-zero for all.

            // we treat zero-blocks as flat for later test.
            n_flat += n_zero;
        }
        else
        {
            // signal block significance.
            printf("sig1 %s\n", hls_sig_update ? "(update)" : "(noupdate)");
            layer_BAC.encodeBinEP(1);
            auto ctx = BinProbModel_Std(PROBA_50_STATE);
            for (int by = 0; by < nby; by++)
            {
                for (int bx = 0; bx < nbx; bx++)
                {
                    if (hls_sig_update)
                        layer_BAC.encodeBin(ctx, !!blk_sig[by*nbx+bx], true);
                    else
                        layer_BAC.encodeBinEP(!!blk_sig[by*nbx+bx]);
                }
            }

#if ENTROPY_CODING_DEBUG
            if (hls_sig_update)
                printf("sigupdate cnt=%d bits=%d\n", ctx.m_binCnt, ctx.m_numBits);
#endif
        }


        // FLAT?
        printf("nflat %d vs %d\n", n_flat, nby*nbx);
        if (n_flat <= nby*nbx/20)
        {
            layer_BAC.encodeBinEP(0);
            memset(blk_flat, 0, nby*nbx*sizeof(blk_flat[0]));
        }
        else
        {
            layer_BAC.encodeBinEP(1);
            // signal flat for sig blocks.
            auto ctx = BinProbModel_Std(PROBA_50_STATE);
            printf("flat1\n");
            for (int by = 0; by < nby; by++)
            {
                for (int bx = 0; bx < nbx; bx++)
                {
                    //if (!blk_sig[by*nbx+bx])
                    //    printf(" ");
                    //else
                    //    printf("%d", !!blk_flat[by*nbx+bx]);
                    if (blk_sig[by*nbx+bx])
                    {
                        if (hls_sig_update)
                            layer_BAC.encodeBin(ctx, !!blk_flat[by*nbx+bx], true);
                        else
                            layer_BAC.encodeBinEP(!!blk_flat[by*nbx+bx]);
                    }
                }
            }
#if ENTROPY_CODING_DEBUG
            if (hls_sig_update)
                printf("flatupdate cnt=%d bits=%d\n", ctx.m_binCnt, ctx.m_numBits);
#endif
        }
    }

    // LATENTS
    for (int y = 0; y < layer_height; y++)
    {
    for (int x = 0; x < layer_width; x++)
    {
        if (hls_sig_blksize > 0 && !blk_sig[(y>>hls_sig_shift)*nbx+(x>>hls_sig_shift)])
            continue;
        if (hls_sig_blksize > 0 && blk_flat[(y>>hls_sig_shift)*nbx+(x>>hls_sig_shift)]
                                && (y%hls_sig_blksize != 0 || x%hls_sig_blksize != 0))
        {
            // in a flat (same-valued) block, and not the 1st pixel.
            continue;
        }

        int idx = y*layer_width+x;
        int val_to_code = xs[idx];
        int val_mu = mus[idx];
        int val_log_sig = log_scales[idx];

        int val_mu_rounded;
        int val_mu_index;
        int val_log_sig_index;
        get_val_mu_indicies(val_mu, val_log_sig, val_mu_rounded, val_mu_index, val_log_sig_index);
        val_to_code = val_to_code - val_mu_rounded;

        auto coding_ctxs = &g_contexts[val_mu_index][val_log_sig_index];
        code_val(layer_BAC, coding_ctxs, val_to_code);
    }
    }
    fflush(stdout);

    layer_BAC.encodeBinTrm(1);
    layer_BAC.finish();
    bsBAC.write(1, 1);
    bsBAC.writeAlignZero();

    // write.
    std::vector<unsigned char> &fifo = bsBAC.getFifo();
    FILE *fout = fopen(out_name.c_str(), "wb");
    if (fout == NULL)
    {
        printf("Cannot open %s for writing\n", out_name.c_str());
        exit(1);
    }
    if (fwrite(fifo.data(), fifo.size(), 1, fout) != 1)
    {
        printf("Write failure to %s\n", out_name.c_str());
        exit(1);
    }
    fclose(fout);
    printf("%s created\n", out_name.c_str());

    delete[] blk_sig;
    delete[] blk_flat;

#if ENTROPY_CODING_DEBUG
    printf("LAYER STATS\n");
    printf("epbits: %d\n", g_epbits);
    printf("trmbits: %d\n", g_trmbits);
    g_epbits = 0;
    g_trmbits = 0;
    for (int m = 0; m < N_MUQ+1; m++)
    {
        for (int s = 0; s < N_SIGQ; s++)
        {
            if (g_contexts[m][s].m_gt0.m_binCnt == 0)
                continue;
            printf("mu[%d]sig[%d] \"gt0\" idx=%d cnt=%d 0s=%d bits=%d\n", m, s, g_contexts[m][s].m_gt0.m_stateIdx, g_contexts[m][s].m_gt0.m_binCnt, g_contexts[m][s].m_gt0.m_zeroCnt, g_contexts[m][s].m_gt0.m_numBits);
            printf("mu[%d]sig[%d] \"gt1\" idx=%d cnt=%d 0s=%d bits=%d\n", m, s, g_contexts[m][s].m_gt1.m_stateIdx, g_contexts[m][s].m_gt1.m_binCnt, g_contexts[m][s].m_gt1.m_zeroCnt, g_contexts[m][s].m_gt1.m_numBits);
            printf("mu[%d]sig[%d] \"gt2\" idx=%d cnt=%d 0s=%d bits=%d\n", m, s, g_contexts[m][s].m_gt2.m_stateIdx, g_contexts[m][s].m_gt2.m_binCnt, g_contexts[m][s].m_gt2.m_zeroCnt, g_contexts[m][s].m_gt2.m_numBits);
            printf("mu[%d]sig[%d] \"gt3\" idx=%d cnt=%d 0s=%d bits=%d\n", m, s, g_contexts[m][s].m_gt3.m_stateIdx, g_contexts[m][s].m_gt3.m_binCnt, g_contexts[m][s].m_gt3.m_zeroCnt, g_contexts[m][s].m_gt3.m_numBits);
            printf("mu[%d]sig[%d] \"pos\" idx=%d cnt=%d 0s=%d bits=%d\n", m, s, g_contexts[m][s].m_ppos.m_stateIdx, g_contexts[m][s].m_ppos.m_binCnt, g_contexts[m][s].m_ppos.m_zeroCnt, g_contexts[m][s].m_ppos.m_numBits);
            g_contexts[m][s].m_gt0.m_binCnt = 0; g_contexts[m][s].m_gt0.m_zeroCnt = 0; g_contexts[m][s].m_gt0.m_numBits = 0;
            g_contexts[m][s].m_gt1.m_binCnt = 0; g_contexts[m][s].m_gt1.m_zeroCnt = 0; g_contexts[m][s].m_gt1.m_numBits = 0;
            g_contexts[m][s].m_gt2.m_binCnt = 0; g_contexts[m][s].m_gt2.m_zeroCnt = 0; g_contexts[m][s].m_gt2.m_numBits = 0;
            g_contexts[m][s].m_gt3.m_binCnt = 0; g_contexts[m][s].m_gt3.m_zeroCnt = 0; g_contexts[m][s].m_gt3.m_numBits = 0;
            g_contexts[m][s].m_ppos.m_binCnt = 0; g_contexts[m][s].m_ppos.m_zeroCnt = 0; g_contexts[m][s].m_ppos.m_numBits = 0;
        }
    }
#endif
}

cc_decode_wb::cc_decode_wb(const std::string &in_file)
{
    FILE *fin = fopen(in_file.c_str(), "rb");
    if (fin == 0)
    {
        printf("Cannot open %s for reading\n", in_file.c_str());
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    long fsz = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    std::vector<unsigned char> file_data(fsz);
    if (fread((char *)file_data.data(), fsz, 1, fin) != 1)
    {
        printf("short read from %s\n", in_file.c_str());
        fclose(fin);
        exit(1);
    }

    fclose(fin);

    m_bs.getFifo() = file_data;
    m_cabac.init(&m_bs);
    m_cabac.start();
}

std::vector<int> cc_decode_wb::decode_wb_continue(int n_weights, int count)
{
    std::vector<int> result(n_weights);

    for (int i = 0; i < n_weights; i++)
    {
        int val = m_cabac.decodeExGolomb(count);
        if (val != 0)
        {
            if (m_cabac.decodeBinEP() != 0)
                val = -val;
        }
        result[i] = val;
    }
    return result;
}

