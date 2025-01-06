/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

// some numbers and indices related to mu and sig quantization.
int const N_MUQ = 16;  // number of mu offsets.
int const N_SIGQ = 50; // number of sig values. now 50, so multiple of 10
int const ZERO_MU = N_MUQ/2;

int const SIG_LOG_MIN = -1; // this min is IN the set.
int const SIG_LOG_MAX_EXCL = 9; // this max is NOT in the set.

int const PROBA_50_STATE = (2*32+1); // generate a BinProbModel_Std with 50% probability.

inline
void get_val_mu_indicies(int val_mu, int val_log_sig,
                         int &r_val_mu_rounded, int &r_val_mu_index, int &r_val_log_sig_index)
{
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
    {
        val_log_sig_index = val_log_sig*(N_SIGQ/(SIG_LOG_MAX_EXCL-SIG_LOG_MIN))+ARM_SCALE/2;
        val_log_sig_index >>= ARM_PRECISION;
        if (val_log_sig_index >= N_SIGQ)
            val_log_sig_index = N_SIGQ-1;
    }

    r_val_mu_rounded = val_mu_rounded>>ARM_PRECISION;
    r_val_mu_index = val_mu_index;
    r_val_log_sig_index = val_log_sig_index;
}


// contexts 17 mus, 50 sigmas
// Context numbers for gtx and ppos for a given mu and sigma.
class MuSigGTs
{
public:
    MuSigGTs(int gt0, int gt1, int gt2, int gt3, int ppos)
    {
        m_gt0 = BinProbModel_Std(gt0);
        m_gt1 = BinProbModel_Std(gt1);
        m_gt2 = BinProbModel_Std(gt2);
        m_gt3 = BinProbModel_Std(gt3);
        m_ppos = BinProbModel_Std(ppos);
    }
    ~MuSigGTs() {}
public:
    BinProbModel_Std m_gt0;
    BinProbModel_Std m_gt1;
    BinProbModel_Std m_gt2;
    BinProbModel_Std m_gt3;
    BinProbModel_Std m_ppos;
};

extern MuSigGTs g_contexts[N_MUQ+1][N_SIGQ];
