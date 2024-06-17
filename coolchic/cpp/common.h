/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#include <memory.h>

int const ARM_MAX_N_FEATURES = 32;

// 64-alignment for possible avx512 use
#define ALIGNTO 64
#define ALIGN(val) ((val+ALIGNTO-1)/ALIGNTO*ALIGNTO)

int const Q_STEP_ARM_WEIGHT_SHIFT[] = {
    8, // 1.0/(1<<8),
    7, // 1.0/(1<<7),
    6, // 1.0/(1<<6),
    5, // 1.0/(1<<5),
    4, // 1.0/(1<<4),
    3, // 1.0/(1<<3),
    2, // 1.0/(1<<2),
    1, // 1.0/(1<<1),
    0 // 1.0/(1<<0),
    };

int const Q_STEP_ARM_BIAS_SHIFT[] = {
    16, // 1.0/(1<<16),
    15, // 1.0/(1<<15),
    14, // 1.0/(1<<14),
    13, // 1.0/(1<<13),
    12, // 1.0/(1<<12),
    11, // 1.0/(1<<11),
    10, // 1.0/(1<<10),
    9,  // 1.0/(1<<9),
    8,  // 1.0/(1<<8),
    7,  // 1.0/(1<<7),
    6,  // 1.0/(1<<6),
    5,  // 1.0/(1<<5),
    4,  // 1.0/(1<<4),
    3,  // 1.0/(1<<3),
    2,  // 1.0/(1<<2),
    1,  // 1.0/(1<<1),
    0   // 1.0/(1<<0),
    };

int const Q_STEP_UPS_SHIFT[] = {
    // 16, // 1.0/(1<<16),
    15, // 1.0/(1<<15),
    14, // 1.0/(1<<14),
    13, // 1.0/(1<<13),
    12, // 1.0/(1<<12),
    11, // 1.0/(1<<11),
    10, // 1.0/(1<<10),
     9, // 1.0/(1<<9),
     8, // 1.0/(1<<8),
     7, // 1.0/(1<<7),
     6, // 1.0/(1<<6),
     5, // 1.0/(1<<5),
     4, // 1.0/(1<<4),
     3, // 1.0/(1<<3),
     2, // 1.0/(1<<2),
     1, // 1.0/(1<<1),
     0, // 1.0/(1<<0),
    };

int const *const Q_STEP_SYN_SHIFT = Q_STEP_UPS_SHIFT;

// the above range is divided into N_SIGQ, so 24/48 (0.5) per entry.
// holds stuff related to our BAC layer decoding.
// basically a bucket.
class BACContext {
public:
    BACContext()
        : m_layer_CABAC(NULL),
          m_layer_height(0),
          m_layer_width(0),
          m_hls_sig_blksize(0),
          m_hls_sig_blksize_shift(0),
          m_hls_sig_blksize_mask(0),
          m_hls_sig_blksize_updated(0),
          m_nby(0),
          m_nbx(0),
          m_blk_sig(NULL),
          m_blk_flat(NULL)
    {
    };

    // set up to decode a layer
    // a grid of significant blocks is read during object creation,
    // avoiding a later decode of any symbols that are to be found
    // in a block with no sigs.
    void set_layer(TDecBinCABAC *layerCABAC, int layer_height, int layer_width, int hls_sig_blksize)
    {
        m_layer_CABAC = layerCABAC;
        m_layer_height = layer_height;
        m_layer_width = layer_width;
        m_hls_sig_blksize_updated = hls_sig_blksize < 0;
        m_hls_sig_blksize = abs(hls_sig_blksize);
        m_hls_sig_blksize_shift = 0;
        while ((1<<m_hls_sig_blksize_shift) < m_hls_sig_blksize)
            m_hls_sig_blksize_shift++;
        m_hls_sig_blksize_mask = (1<<m_hls_sig_blksize_shift)-1;

        if (m_hls_sig_blksize > 0)
        {
            m_nby = (m_layer_height+m_hls_sig_blksize-1)>>m_hls_sig_blksize_shift;
            m_nbx = (m_layer_width+m_hls_sig_blksize-1)>>m_hls_sig_blksize_shift;
        }
        else
        {
            m_nby = 1;
            m_nbx = 1;
        }
        delete[] m_blk_sig;
        m_blk_sig = new unsigned char[m_nby*m_nbx];
        delete[] m_blk_flat;
        m_blk_flat = new unsigned char[m_nby*m_nbx];

        memset(m_blk_sig, 1, m_nby*m_nbx*sizeof(m_blk_sig[0]));
        memset(m_blk_flat, 0, m_nby*m_nbx*sizeof(m_blk_flat[0]));

        if (m_nby != 1 || m_nbx != 1)
        {
            int idx = 0;
            if (m_layer_CABAC->decodeBinEP() != 0)
            {
                // signaled
                //printf("sig:\n");
                if (m_hls_sig_blksize_updated)
                {
                    // updated context: proba 0.5 is raw index 32
                    auto sigctx = BinProbModel_Std(2*32+1);
                    for (int y = 0; y < m_nby; y++)
                    {
                        for (int x = 0; x < m_nbx; x++)
                        {
                            m_blk_sig[idx++] = m_layer_CABAC->decodeBin(sigctx, true);
                            //printf("%c", m_blk_sig[idx-1] ? 'X' : '.');
                        }
                        //printf("\n");
                    }
                }
                else
                {
                    for (int y = 0; y < m_nby; y++)
                    {
                        for (int x = 0; x < m_nbx; x++)
                        {
                            m_blk_sig[idx++] = m_layer_CABAC->decodeBinEP();
                            //printf("%c", m_blk_sig[idx-1] ? 'X' : '.');
                        }
                        //printf("\n");
                    }
                }
            }

            idx = 0;
            if (m_layer_CABAC->decodeBinEP() != 0)
            {
                // signaled
                //printf("flat:\n");
                if (m_hls_sig_blksize_updated)
                {
                    // updated context: proba 0.5 is raw index 32
                    auto sigctx = BinProbModel_Std(2*32+1);
                    for (int y = 0; y < m_nby; y++)
                    {
                        for (int x = 0; x < m_nbx; x++)
                        {
                            if (m_blk_sig[idx])
                            {
                                m_blk_flat[idx] = m_layer_CABAC->decodeBin(sigctx, true);
                                //printf("%c", m_blk_sig[idx] ? 'X' : '.');
                            }
                            idx++;
                        }
                        //printf("\n");
                    }
                }
                else
                {
                    for (int y = 0; y < m_nby; y++)
                    {
                        for (int x = 0; x < m_nbx; x++)
                        {
                            if (m_blk_sig[idx])
                            {
                                m_blk_flat[idx] = m_layer_CABAC->decodeBinEP();
                                //printf("%c", m_blk_sig[idx] ? 'X' : '.');
                            }
                            idx++;
                        }
                        //printf("\n");
                    }
                }
            }
        }
    };

    ~BACContext() { delete[] m_blk_sig; }

public:
    TDecBinCABAC *m_layer_CABAC;
    int m_layer_height;
    int m_layer_width;
    int m_hls_sig_blksize;
    int m_hls_sig_blksize_shift;
    int m_hls_sig_blksize_mask;
    int m_hls_sig_blksize_updated;
    int m_nby;
    int m_nbx;
    unsigned char *m_blk_sig;
    unsigned char *m_blk_flat;
};

inline
bool bac_coded(BACContext &bac_context, int start_y, int start_x)
{
   int const hls_sig_blksize = bac_context.m_hls_sig_blksize;
   int const hls_sig_blksize_shift = bac_context.m_hls_sig_blksize_shift;

   if (hls_sig_blksize == 0)
    return true;

   auto blk_sig = bac_context.m_blk_sig;
   auto nbx = bac_context.m_nbx;

   if (blk_sig[(start_y>>hls_sig_blksize_shift)*nbx+(start_x>>hls_sig_blksize_shift)] == 0)
   {
       return false;
   }
   return true;
}

inline
bool bac_flat(BACContext &bac_context, int start_y, int start_x, int &use_left)
{
    int const hls_sig_blksize_shift = bac_context.m_hls_sig_blksize_shift;
    int const hls_sig_blksize_mask = bac_context.m_hls_sig_blksize_mask;

    auto blk_flat = bac_context.m_blk_flat;
    auto nbx = bac_context.m_nbx;
 
    if (blk_flat[(start_y>>hls_sig_blksize_shift)*nbx+(start_x>>hls_sig_blksize_shift)] == 0)
        return false;
    if ((start_x&hls_sig_blksize_mask) != 0)
    {
        use_left = 1;
        return true;
    }
    else if ((start_y&hls_sig_blksize_mask) != 0)
    {
        use_left = 0;
        return true;
    }
    else
        return false;
}

inline
int32_t decode_single(
           TDecBinCABAC *cabac,
           MuSigGTs *coding_ctxs
           )
{
   // int bits_start = cabac->bitsRead();
   auto gt0 = cabac->decodeBin(coding_ctxs->m_gt0, false);
   int coded_val;
   if (gt0 == 0)
       coded_val = 0;
   else
   {
       // significant
       auto gt1 = cabac->decodeBin(coding_ctxs->m_gt1, false);
       if (gt1 == 0)
           coded_val = 1;
       else
       {
           auto gt2 = cabac->decodeBin(coding_ctxs->m_gt2, false);
           if (gt2 == 0)
               coded_val = 2;
           else
           {
               auto gt3 = cabac->decodeBin(coding_ctxs->m_gt3, false);
               if (gt3 == 0)
                   coded_val = 3;
               else
               {
                   coded_val = cabac->decodeExGolomb(0) + 3 + 1;
               }
           }
       }
       if (cabac->decodeBin(coding_ctxs->m_ppos, false) != 0)
           coded_val = -coded_val;
   }
   // int bits_end = cabac->bitsRead();
   // printf("gt0state %d val %d bits %d\n", coding_ctxs->m_gt0.getStateMps(), coded_val, bits_end-bits_start);
   return coded_val;
}

// raw decode.
int32_t *decode_weights(TDecBinCABAC *cabac, MuSigGTs *coding_ctxs, int n_weights);
// decode with dequant
float *decode_weights_q(TDecBinCABAC *cabac, MuSigGTs *coding_ctxs, int n_weights, int q_step_shift);

// should have called bac_coded prior.  We don't check here.
inline
int32_t decode_latent_layer_bac_single(
           BACContext &bac_context,
           int32_t mu,
           int32_t log_scale
           )
{
   auto layer_BAC = bac_context.m_layer_CABAC;

   int val_mu_rounded;
   int val_mu_index;
   int val_sig_index;

   get_val_mu_indicies(mu, log_scale, //sig_index_from_qlog,
                val_mu_rounded, val_mu_index, val_sig_index);
   auto coding_ctxs = &g_contexts[val_mu_index][val_sig_index];

   //printf("mu%d,logsig%d; indexedmu%d,sig%d:", mu, log_scale, val_mu_index, val_sig_index);
   return val_mu_rounded+decode_single(layer_BAC, coding_ctxs);
}

void arm_scale(int32_t *wb, int n_wb, int q_step_shift);
