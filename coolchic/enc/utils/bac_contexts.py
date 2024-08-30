# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import math  # for math.isnan

import torch
from enc.component.coolchic import _laplace_cdf
from enc.utils.misc import bac_state_idx_from_proba_0

N_MUQ = 16
N_SIGQ = 50  # taking a range of 10 to N_SIGQ - should be an integer multiple.

# Should be in utils/constants.py
SIG_LOG_MIN = -5 + 4  # -4.6
SIG_LOG_MAX_EXCL = 5 + 4


# Probability limits for arithmetic stability and reasonableness
P_MIN = torch.tensor([0.001])
P_MAX = torch.tensor([1 - 0.001])


# Convert proba to something reasonable, ie, not 0 or 1
def reasonable_proba(p):
    p = torch.abs(p)
    if p < P_MIN:
        p = P_MIN
    if p > P_MAX:
        p = P_MAX
    return p


# Generate context-table information, taking [muidx][sigidx] to gt0,gt1,gt2,gt3,ppos states.
def get_contexts(contexts_cpp: str = ""):
    # Get all possible sigmas, quantized prior to the exp.
    inputs = torch.arange(
        SIG_LOG_MIN, SIG_LOG_MAX_EXCL, (SIG_LOG_MAX_EXCL - SIG_LOG_MIN) / N_SIGQ
    )
    # Take log sig identifiers to a spread indicator.
    sigs_quanted = torch.exp(inputs - 4).to("cpu")

    probas = []  # index with [muidx][sigidx] to get {"gt0"...}

    R = torch.tensor([0])  # All our offsets here round to 0.
    mu_min = 0 - N_MUQ // 2
    mu_max = (
        N_MUQ // 2 + 1
    )  # We do an extra element at the end to allow easy switching to -ve mus.
    # eg, [-8..+8] rather than [-8..+7]

    for mu_offset in range(mu_min, mu_max):
        mu_offset = torch.tensor([mu_offset])
        sigs = []
        for sig in sigs_quanted:
            # gtx is the proba of sending a 0 (ie, "not gtx").
            gt0_surface = _laplace_cdf(R + 0.5, mu_offset / N_MUQ, sig) - _laplace_cdf(
                R - 0.5, mu_offset / N_MUQ, sig
            )
            gt0 = gt0_surface / 1.0
            gt0 = reasonable_proba(gt0)
            if gt0 == P_MAX:
                # gt0 is maxxed out due to a spikey sigma.
                # We leave gt0 at MAX
                gt1 = torch.tensor([0.5])
                gt2 = torch.tensor([0.5])
                gt3 = torch.tensor([0.5])
            else:
                gt1_surface = (
                    _laplace_cdf(R + 1 + 0.5, mu_offset / N_MUQ, sig)
                    - _laplace_cdf(R + 1 - 0.5, mu_offset / N_MUQ, sig)
                ) + (
                    _laplace_cdf(R - 1 + 0.5, mu_offset / N_MUQ, sig)
                    - _laplace_cdf(R - 1 - 0.5, mu_offset / N_MUQ, sig)
                )
                if gt1_surface <= P_MIN:
                    # Protect extremely spikey sigma from having no large-valued populations.
                    gt1 = torch.tensor([0.5])
                    gt2 = torch.tensor([0.5])
                    gt3 = torch.tensor([0.5])
                else:
                    gt1 = gt1_surface / (1 - gt0_surface)
                    gt1 = reasonable_proba(gt1)
                    gt2_surface = (
                        _laplace_cdf(R + 2 + 0.5, mu_offset / N_MUQ, sig)
                        - _laplace_cdf(R + 2 - 0.5, mu_offset / N_MUQ, sig)
                    ) + (
                        _laplace_cdf(R - 2 + 0.5, mu_offset / N_MUQ, sig)
                        - _laplace_cdf(R - 2 - 0.5, mu_offset / N_MUQ, sig)
                    )
                    if gt2_surface <= P_MIN:
                        # Protect extremely spikey sigma from having no large-valued populations.
                        gt2 = torch.tensor([0.5])
                        gt3 = torch.tensor([0.5])
                    else:
                        gt2 = gt2_surface / (1 - gt0_surface - gt1_surface)
                        gt2 = reasonable_proba(gt2)
                        gt3_surface = (
                            _laplace_cdf(R + 3 + 0.5, mu_offset / N_MUQ, sig)
                            - _laplace_cdf(R + 3 - 0.5, mu_offset / N_MUQ, sig)
                        ) + (
                            _laplace_cdf(R - 3 + 0.5, mu_offset / N_MUQ, sig)
                            - _laplace_cdf(R - 3 - 0.5, mu_offset / N_MUQ, sig)
                        )
                        if gt3_surface <= P_MIN:
                            # Protect extremely spikey sigma from having no large-valued populations.
                            gt3 = torch.tensor([0.5])
                        else:
                            gt3 = gt3_surface / (
                                1 - gt0_surface - gt1_surface - gt2_surface
                            )
                            gt3 = reasonable_proba(gt3)

            # proba of sending a 0, indicating +ve
            pos_surface = 1.0 - _laplace_cdf(R + 0.5, mu_offset / N_MUQ, sig)
            neg_surface = _laplace_cdf(R - 0.5, mu_offset / N_MUQ, sig)
            if pos_surface <= P_MIN and neg_surface <= P_MIN:
                ppos = torch.tensor([0.5])
            elif pos_surface <= P_MIN:
                ppos = torch.tensor([0])
            elif neg_surface <= P_MIN:
                ppos = torch.tensor([1])
            else:  # pos_surface >= P_MIN and neg_surface >= P_MIN:
                ppos = pos_surface / (pos_surface + neg_surface)
            ppos = reasonable_proba(ppos)

            these_probas = {
                "gt0": gt0,
                "gt1": gt1,
                "gt2": gt2,
                "gt3": gt3,
                "ppos": ppos,
            }
            if (
                math.isnan(gt0)
                or math.isnan(gt1)
                or math.isnan(gt2)
                or math.isnan(gt3)
                or math.isnan(ppos)
                or gt0 < 0
                or gt1 < 0
                or gt2 < 0
                or gt3 < 0
                or ppos < 0
            ):
                print("NAN in table!")
                print("mu_offset", mu_offset, "sig", sig, "idx", len(probas))
                print(these_probas)
                exit(1)
            sigs.append(these_probas)
        probas.append(sigs)

    # Convert these probas to (CA)BAC contexts. We get the closest available context.
    contexts = []
    for sigs in probas:
        sig_ctxs = []
        for ps in sigs:
            # Convert p(0) to a context state
            gt0 = bac_state_idx_from_proba_0(ps["gt0"])
            gt1 = bac_state_idx_from_proba_0(ps["gt1"])
            gt2 = bac_state_idx_from_proba_0(ps["gt2"])
            gt3 = bac_state_idx_from_proba_0(ps["gt3"])
            ppos = bac_state_idx_from_proba_0(ps["ppos"])
            these_ctxs = {"gt0": gt0, "gt1": gt1, "gt2": gt2, "gt3": gt3, "ppos": ppos}
            sig_ctxs.append(these_ctxs)
        contexts.append(sig_ctxs)

    if contexts_cpp != "":
        # print out the contexts to .h and .cpp files, allowing decoding from c++.
        with open(contexts_cpp + ".h", "wt") as f:
            print(
                f"""
/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

// some numbers and indices related to mu and sig quantization.
int const N_MUQ = {N_MUQ};  // number of mu offsets.
int const N_SIGQ = {N_SIGQ}; // number of sig values. now 50, so multiple of 10
int const ZERO_MU = N_MUQ/2;

int const SIG_LOG_MIN = {SIG_LOG_MIN}; // this min is IN the set.
int const SIG_LOG_MAX_EXCL = {SIG_LOG_MAX_EXCL}; // this max is NOT in the set.

int const PROBA_50_STATE = (2*32+1); // generate a BinProbModel_Std with 50% probability.

inline
void get_val_mu_indicies(int val_mu, int val_log_sig,
                         int &r_val_mu_rounded, int &r_val_mu_index, int &r_val_log_sig_index)
{{
    int val_mu_rounded = val_mu;
    val_mu_rounded = (val_mu_rounded >= 0) ? (val_mu_rounded+ARM_SCALE/2)>>ARM_PRECISION<<ARM_PRECISION : -((-val_mu_rounded+ARM_SCALE/2)>>ARM_PRECISION<<ARM_PRECISION);

    int val_mu_index = (val_mu - val_mu_rounded)*N_MUQ;
    // round to an index
    val_mu_index = val_mu_index >= 0 ? ((val_mu_index+ARM_SCALE/2)>>ARM_PRECISION) : -((-val_mu_index+ARM_SCALE/2)>>ARM_PRECISION);
    val_mu_index += N_MUQ/2;

    // no longer a table.
    int val_log_sig_index;
    val_log_sig -= SIG_LOG_MIN*ARM_SCALE;
    if (val_log_sig < 0)
        val_log_sig_index = 0;
    else
    {{
        val_log_sig_index = val_log_sig*(N_SIGQ/(SIG_LOG_MAX_EXCL-SIG_LOG_MIN))+ARM_SCALE/2;
        val_log_sig_index >>= ARM_PRECISION;
        if (val_log_sig_index >= N_SIGQ)
            val_log_sig_index = N_SIGQ-1;
    }}

    r_val_mu_rounded = val_mu_rounded>>ARM_PRECISION;
    r_val_mu_index = val_mu_index;
    r_val_log_sig_index = val_log_sig_index;
}}


// contexts {len(contexts)} mus, {len(contexts[0])} sigmas
// Context numbers for gtx and ppos for a given mu and sigma.
class MuSigGTs
{{
public:
    MuSigGTs(int gt0, int gt1, int gt2, int gt3, int ppos)
    {{
        m_gt0 = BinProbModel_Std(gt0);
        m_gt1 = BinProbModel_Std(gt1);
        m_gt2 = BinProbModel_Std(gt2);
        m_gt3 = BinProbModel_Std(gt3);
        m_ppos = BinProbModel_Std(ppos);
    }}
    ~MuSigGTs() {{}}
public:
    BinProbModel_Std m_gt0;
    BinProbModel_Std m_gt1;
    BinProbModel_Std m_gt2;
    BinProbModel_Std m_gt3;
    BinProbModel_Std m_ppos;
}};

extern MuSigGTs g_contexts[N_MUQ+1][N_SIGQ];""",
                file=f,
            )

        with open(contexts_cpp + ".cpp", "wt") as f:
            print(
                f"""
/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#include "Contexts.h"
#include "common.h"
#include "cc-contexts.h"

MuSigGTs g_contexts[N_MUQ+1][N_SIGQ] = {{""",
                file=f,
            )

            for mu_idx in range(len(contexts)):
                print("{", file=f)
                for sig_idx in range(len(contexts[mu_idx])):
                    ctxs = contexts[mu_idx][sig_idx]
                    print(
                        "  MuSigGTs( %d,%d,%d,%d,%d ),"
                        % (
                            ctxs["gt0"],
                            ctxs["gt1"],
                            ctxs["gt2"],
                            ctxs["gt3"],
                            ctxs["ppos"],
                        ),
                        file=f,
                    )
                print("},", file=f)

            print("};", file=f)

        print(contexts_cpp + ".h", "created")
        print(contexts_cpp + ".cpp", "created")

    return contexts, sigs_quanted, probas  # probas is only for estimation stats


## Given ARM outputs mu and log_scale, return indicies into MU and SIGMA tables.
# def get_val_mu_indices(
#        val_mu,
#        val_log_sig,
#        sigs_quanted = None
#    ):
#    # !!! can use ints after *64.
#    if val_mu >= 0:
#        val_mu_rounded = (val_mu+0.5).to(torch.int32)
#    else:
#        val_mu_rounded = -(-val_mu+0.5).to(torch.int32)
#    #print("oldrnd", val_mu_rounded)
#    val_mu_index = (val_mu - val_mu_rounded.to(torch.float))*N_MUQ
#    #print("old1", val_mu_index)
#    if val_mu_index >= 0:
#        val_mu_index = (val_mu_index+0.5).to(torch.int32)
#    else:
#        val_mu_index = -(-val_mu_index+0.5).to(torch.int32)
#    #print("old2", val_mu_index)
#    val_mu_index += N_MUQ//2 # min of -8 goes to 0
#
#    # val_sig_index = torch.round(val_sig*N_SIGQ).to(torch.int32).clamp(min=sigminQ)-sigminQ
#    val_log_sig_index = val_log_sig-SIG_LOG_MIN
#    #print("oldsig1", val_log_sig_index)
#    val_log_sig_index = val_log_sig_index*(N_SIGQ//(SIG_LOG_MAX_EXCL-SIG_LOG_MIN))
#    #print("oldsig2", val_log_sig_index)
#    if val_log_sig_index < 0:
#        val_log_sig_index = -(-val_log_sig_index+0.5).to(torch.int32)
#    else:
#        val_log_sig_index = (val_log_sig_index+0.5).to(torch.int32)
#    if val_log_sig_index < 0:
#        val_log_sig_index = 0
#    elif val_log_sig_index >= N_SIGQ:
#        val_log_sig_index = N_SIGQ-1
#
#    old_val_mu_index = val_mu_index
#    old_val_log_sig_index = val_log_sig_index
#
#    # !!! can use ints after *64.
#    val_mu64 = torch.round(val_mu*64).to(torch.int32)
#    val_log_sig64 = torch.round(val_log_sig*64).to(torch.int32)
#
#    if val_mu64 >= 0:
#        val_mu_rounded64 = ((val_mu64+32)>>6)<<6
#    else:
#        val_mu_rounded64 = -(((-val_mu64+32)>>6)<<6)
#    #print("newrnd", val_mu_rounded64)
#    val_mu_index = ((val_mu64 - val_mu_rounded64)*N_MUQ)
#    if val_mu_index < 0:
#        val_mu_index = -((-val_mu_index+32)//64)
#    else:
#        val_mu_index = (val_mu_index+32)//64
#    #print("new1", val_mu_index)
#    #if val_mu_index >= 0:
#    #    val_mu_index = (val_mu_index+0.5).to(torch.int32)
#    #else:
#    #    val_mu_index = -(-val_mu_index+0.5).to(torch.int32)
#    val_mu_index += N_MUQ//2 # min of -8 goes to 0
#
#    # val_sig_index = torch.round(val_sig*N_SIGQ).to(torch.int32).clamp(min=sigminQ)-sigminQ
#    val_log_sig_index = val_log_sig64-SIG_LOG_MIN*64
#    #print("newsig1", val_log_sig_index)
#    val_log_sig_index = val_log_sig_index*(N_SIGQ//(SIG_LOG_MAX_EXCL-SIG_LOG_MIN))
#    #print("newsig2", val_log_sig_index)
#    if val_log_sig_index < 0:
#        val_log_sig_index = -((-val_log_sig_index+32)//64)
#    else:
#        val_log_sig_index = (val_log_sig_index+32)//64
#    #val_log_sig_index = (val_log_sig_index+0.5).to(torch.int32)
#    if val_log_sig_index < 0:
#        val_log_sig_index = 0
#    elif val_log_sig_index >= N_SIGQ:
#        val_log_sig_index = N_SIGQ-1
#
#    if val_mu_index != old_val_mu_index:
#        print("problem mu", val_mu_index, "!= old", old_val_mu_index, "from val", val_mu, val_mu*64)
#        exit(1)
#    if val_log_sig_index != old_val_log_sig_index:
#        print("problem sig", val_log_sig_index, "!= old", old_val_log_sig_index, "from val", val_log_sig, val_log_sig*64)
#        exit(1)
#    #print("ok")
#
#    # val_sig_index = torch.argmin((sigs_quanted - val_sig).abs()).item()
#    #print("val_mu", val_mu, "(index", val_mu_index, ")  val_log_sig", val_log_sig, "(index", val_log_sig_index, ")")
#    # print("sigs_quanted", sigs_quanted)
#    #print(val_sig, "found at index", val_sig_index)
#    if val_mu_index < 0 or val_mu_index > N_MUQ:
#        print()
#        print("bad muidx")
#        print("mu", val_mu, "rounded", val_mu_rounded, "muidx", val_mu_index, flush=True)
#        print("orig offs", ((val_mu - val_mu_rounded.to(torch.float))*N_MUQ).to(torch.int32))
#        print("after adding", N_MUQ//2, "=", val_mu_index, flush=True)
#        exit(1)
#
#    return val_mu_rounded, val_mu_index, val_log_sig_index
#
# def decode_latent_layer_bac_init(
#                bitstream_bytes: bytes,
#                layer_height: int,
#                layer_width: int,
#                hls_sig_blksize: int,
#                ):
#    #with open(bitstream_path, "rb") as f:
#    #    data = f.read()
#
#    contexts, sigs_quanted, probas = get_contexts()
#    layer_BAC = CabacDecoder(bitstream_bytes)
#
#    # Get block significance.
#    if hls_sig_blksize > 0:
#        nby = (layer_height+hls_sig_blksize-1)//hls_sig_blksize
#        nbx = (layer_width+hls_sig_blksize-1)//hls_sig_blksize
#    else:
#        nby = 1
#        nbx = 1
#    blk_sig = torch.ones(nby, nbx)
#    blk_flat = torch.zeros(nby, nbx)
#    if nby != 1 or nbx != 1:
#        # Using significance map?
#        use_sig = layer_BAC.decodeBinEP()
#        if use_sig:
#            for by in range(0, nby):
#                for bx in range(0, nbx):
#                    blk_sig[by, bx] = layer_BAC.decodeBinEP()
#        # Using zero-residue (flat) map?
#        use_flat = layer_BAC.decodeBinEP()
#        if use_flat:
#            for by in range(0, nby):
#                for bx in range(0, nbx):
#                    if blk_sig[by, bx]:
#                        blk_flat[by, bx] = layer_BAC.decodeBinEP()
#
#    print("blk_sig", blk_sig, flush=True)
#    print("blk_flat", blk_flat, flush=True)
#    context = {
#        'contexts': contexts,
#        'layer_width': layer_width,
#        'layer_height': layer_height,
#        'sigs_quanted': sigs_quanted,
#        'probas': probas,
#        'layer_BAC': layer_BAC,
#        'blk_sig': blk_sig,
#        'blk_flat': blk_flat,
#        'hls_sig_blksize': hls_sig_blksize,
#        }
#    return context
#
## Now takes log_scale directly from arm.
# def decode_latent_layer_bac_continue(
#            bac_context,
#            current_y, pad, # flat-block neighbor pickup.
#            indices, # wavefront order
#            mus,
#            log_scales
#        ):
#    result = []
#
#    layer_height = bac_context['layer_height']
#    layer_width = bac_context['layer_width']
#    blk_sig = bac_context['blk_sig']
#    blk_flat = bac_context['blk_flat']
#    hls_sig_blksize = bac_context['hls_sig_blksize']
#    # sigs_quanted = bac_context['sigs_quanted']
#    # print("decoding", len(indices), "pixels")
#
#    # encode 10k 0s.
#    #print("decoding 10k 0s at high-0 proba")
#    #ctx = CabacContext.fromproba0(0.999) # Convert p(0) to a context.
#    #layer_BAC = bac_context['layer_BAC']
#    #for xxxx in range(0,10000):
#    #    bit = layer_BAC.decodeBin(ctx)
#    #    print(bit, end="")
#    #print()
#    #print("continuing")
#    #exit(0)
#
#    for idx in range(len(indices)):
#        pixel_idx = indices[idx]
#        xidx = pixel_idx%layer_width
#        yidx = pixel_idx//layer_width
#        if hls_sig_blksize > 0 and blk_sig[yidx//hls_sig_blksize, xidx//hls_sig_blksize] == 0:
#            # print("bac-skip: y%d,x%d"%(yidx, xidx))
#            result.append(0)
#            continue
#        if hls_sig_blksize > 0 and blk_flat[yidx//hls_sig_blksize, xidx//hls_sig_blksize]:
#            if xidx%hls_sig_blksize != 0:
#                # take from left
#                result.append(current_y[(yidx+pad)*(layer_width+2*pad)+xidx+pad-1])
#                continue
#            elif yidx%hls_sig_blksize != 0:
#                # take from up
#                result.append(current_y[(yidx+pad-1)*(layer_width+2*pad)+xidx+pad])
#                continue
#
#        layer_BAC = bac_context['layer_BAC']
#        contexts = bac_context['contexts']
#
#        val_mu_rounded, val_mu_index, val_log_sig_index = get_val_mu_indices(mus[idx], log_scales[idx])
#        # print("mu_index", val_mu_index, "log_sig_index", val_log_sig_index, "mu rounded", val_mu_rounded)
#        coding_ctxs = contexts[val_mu_index][val_log_sig_index]
#
#        gt0 = layer_BAC.decodeBin(coding_ctxs["gt0"])
#        #print("gt0", gt0, end="")
#        if gt0 == 0:
#            coded_val = 0
#        else:
#            # significant
#            gt1 = layer_BAC.decodeBin(coding_ctxs["gt1"])
#            #print(" gt1", gt1, end="")
#            if gt1 == 0:
#                coded_val = 1
#            else:
#                gt2 = layer_BAC.decodeBin(coding_ctxs["gt2"])
#                #print(" gt2", gt2, end="")
#                if gt2 == 0:
#                    coded_val = 2
#                else:
#                    gt3 = layer_BAC.decodeBin(coding_ctxs["gt3"])
#                    #print(" gt3", gt3, end="")
#                    if gt3 == 0:
#                        coded_val = 3
#                    else:
#                        coded_val = layer_BAC.decodeExGolomb(0) + 3 + 1
#                        #print(" exgto", coded_val, end="")
#
#            if layer_BAC.decodeBin(coding_ctxs["ppos"]) != 0:
#                coded_val = -coded_val
#        #print()
#        result.append(val_mu_rounded+coded_val)
#        # print("baccode: y%d,x%d: %d"%(yidx, xidx, val_mu_rounded+coded_val), flush=True)
#
#    result = torch.tensor(result, dtype=torch.float32)
#    #print("decoded x wave", result)
#    return result
