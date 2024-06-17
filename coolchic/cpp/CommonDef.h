/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2017, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     CommonDef.h
    \brief    Defines version information, constants and small in-line functions
*/

#ifndef __COMMONDEF__
#define __COMMONDEF__

#include <algorithm>
#include <iostream>
#include <assert.h>
#include <limits>

#if _MSC_VER > 1000
// disable "signed and unsigned mismatch"
#pragma warning( disable : 4018 )
// disable Bool coercion "performance warning"
#pragma warning( disable : 4800 )
#endif // _MSC_VER > 1000
#include "TypeDef.h"

#ifdef _MSC_VER
#if _MSC_VER <= 1500
inline Int64 abs (Int64 x) { return _abs64(x); };
#endif
#endif

//! \ingroup TLibCommon
//! \{

// ====================================================================================================================
// Version information
// ====================================================================================================================

#define NV_VERSION        "16.18"                 ///< Current software version

// ====================================================================================================================
// Platform information
// ====================================================================================================================

#ifdef __GNUC__
#define NVM_COMPILEDBY  "[GCC %d.%d.%d]", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__
#ifdef __IA64__
#define NVM_ONARCH    "[on 64-bit] "
#else
#define NVM_ONARCH    "[on 32-bit] "
#endif
#endif

#ifdef __INTEL_COMPILER
#define NVM_COMPILEDBY  "[ICC %d]", __INTEL_COMPILER
#elif  _MSC_VER
#define NVM_COMPILEDBY  "[VS %d]", _MSC_VER
#endif

#ifndef NVM_COMPILEDBY
#define NVM_COMPILEDBY "[Unk-CXX]"
#endif

#ifdef _WIN32
#define NVM_ONOS        "[Windows]"
#elif  __linux
#define NVM_ONOS        "[Linux]"
#elif  __CYGWIN__
#define NVM_ONOS        "[Cygwin]"
#elif __APPLE__
#define NVM_ONOS        "[Mac OS X]"
#else
#define NVM_ONOS "[Unk-OS]"
#endif

#define NVM_BITS          "[%d bit] ", (sizeof(Void*) == 8 ? 64 : 32) ///< used for checking 64-bit O/S

#ifndef NULL
#define NULL              0
#endif

// ====================================================================================================================
// Common constants
// ====================================================================================================================

static const UInt   MAX_UINT =                            0xFFFFFFFFU; ///< max. value of unsigned 32-bit integer
static const Int    MAX_INT =                              2147483647; ///< max. value of signed 32-bit integer
static const Double MAX_DOUBLE =                             1.7e+308; ///< max. value of Double-type value

// ====================================================================================================================
// Coding tool configuration
// ====================================================================================================================
// Most of these should not be changed - they resolve the meaning of otherwise magic numbers.

static const Int MAX_GOP =                                         64; ///< max. value of hierarchical GOP size
static const Int MAX_NUM_REF_PICS =                                16; ///< max. number of pictures used for reference
static const Int MAX_NUM_REF =                                     16; ///< max. number of entries in picture reference list
static const Int MAX_QP =                                          51;
static const Int NOT_VALID =                                       -1;

static const Int AMVP_MAX_NUM_CANDS =                               2; ///< AMVP: advanced motion vector prediction - max number of final candidates
static const Int AMVP_DECIMATION_FACTOR =                           4;
static const Int MRG_MAX_NUM_CANDS =                                5; ///< MERGE


static const Int MAX_TLAYER =                                       7; ///< Explicit temporal layer QP offset - max number of temporal layer

static const Int ADAPT_SR_SCALE =                                   1; ///< division factor for adaptive search range

static const Int MAX_NUM_PICS_IN_SOP =                           1024;

static const Int MAX_NESTING_NUM_OPS =                           1024;
static const Int MAX_NESTING_NUM_LAYER =                           64;

static const Int MAX_VPS_NUM_HRD_PARAMETERS =                       1;
static const Int MAX_VPS_OP_SETS_PLUS1 =                         1024;
static const Int MAX_VPS_NUH_RESERVED_ZERO_LAYER_ID_PLUS1 =         1;

static const Int MAXIMUM_INTRA_FILTERED_WIDTH =                    16;
static const Int MAXIMUM_INTRA_FILTERED_HEIGHT =                   16;

static const Int MAX_CPB_CNT =                                     32; ///< Upper bound of (cpb_cnt_minus1 + 1)
static const Int MAX_NUM_LAYER_IDS =                               64;

static const Int COEF_REMAIN_BIN_REDUCTION =                        3; ///< indicates the level at which the VLC transitions from Golomb-Rice to TU+EG(k)

static const Int CU_DQP_TU_CMAX =                                   5; ///< max number bins for truncated unary
static const Int CU_DQP_EG_k =                                      0; ///< expgolomb order

static const Int SBH_THRESHOLD =                                    4; ///< value of the fixed SBH controlling threshold

static const Int C1FLAG_NUMBER =                                    8; // maximum number of largerThan1 flag coded in one chunk:  16 in HM5
static const Int C2FLAG_NUMBER =                                    1; // maximum number of largerThan2 flag coded in one chunk:  16 in HM5

static const Int MAX_NUM_VPS =                                     16;
static const Int MAX_NUM_SPS =                                     16;
static const Int MAX_NUM_PPS =                                     64;


static const Int MLS_GRP_NUM =                                     64; ///< Max number of coefficient groups, max(16, 64)
static const Int MLS_CG_LOG2_WIDTH =                                2;
static const Int MLS_CG_LOG2_HEIGHT =                               2;
static const Int MLS_CG_SIZE =                                      4; ///< Coefficient group size of 4x4; = MLS_CG_LOG2_WIDTH + MLS_CG_LOG2_HEIGHT

#if ADAPTIVE_QP_SELECTION
static const Int ARL_C_PRECISION =                                  7; ///< G382: 7-bit arithmetic precision
static const Int LEVEL_RANGE =                                     30; ///< G382: max coefficient level in statistics collection
#endif

static const Int RVM_VCEGAM10_M =                                   4;

static const Int FAST_UDI_MAX_RDMODE_NUM =                         35; ///< maximum number of RD comparison in fast-UDI estimation loop

static const Int NUM_INTRA_MODE =                                  36;
static const Int PLANAR_IDX =                                       0;
static const Int VER_IDX =                                         26; ///< index for intra VERTICAL   mode
static const Int HOR_IDX =                                         10; ///< index for intra HORIZONTAL mode
static const Int DC_IDX =                                           1; ///< index for intra DC mode
static const Int NUM_CHROMA_MODE =                                  5; ///< total number of chroma modes
static const Int DM_CHROMA_IDX =                                   36; ///< chroma mode index for derived from luma intra mode

static const Int MDCS_ANGLE_LIMIT =                                 4; ///< 0 = Horizontal/vertical only, 1 = Horizontal/vertical +/- 1, 2 = Horizontal/vertical +/- 2 etc...
static const Int MDCS_MAXIMUM_WIDTH =                               8; ///< (measured in pixels) TUs with width greater than this can only use diagonal scan
static const Int MDCS_MAXIMUM_HEIGHT =                              8; ///< (measured in pixels) TUs with height greater than this can only use diagonal scan


static const Int LOG2_MAX_NUM_COLUMNS_MINUS1 =                      7;
static const Int LOG2_MAX_NUM_ROWS_MINUS1 =                         7;

static const Int CABAC_INIT_PRESENT_FLAG =                          1;

static const Int LUMA_INTERPOLATION_FILTER_SUB_SAMPLE_POSITIONS =   4;
static const Int CHROMA_INTERPOLATION_FILTER_SUB_SAMPLE_POSITIONS = 8;

static const Int MAX_NUM_LONG_TERM_REF_PICS =                      33;
static const Int NUM_LONG_TERM_REF_PIC_SPS =                        0;


static const Int MAX_QP_OFFSET_LIST_SIZE =                          6; ///< Maximum size of QP offset list is 6 entries

// Cost mode support
static const Int LOSSLESS_AND_MIXED_LOSSLESS_RD_COST_TEST_QP =      0; ///< QP to use for lossless coding.
static const Int LOSSLESS_AND_MIXED_LOSSLESS_RD_COST_TEST_QP_PRIME =4; ///< QP' to use for mixed_lossy_lossless coding.

static const Int RExt__GOLOMB_RICE_ADAPTATION_STATISTICS_SETS =     4;
static const Int RExt__GOLOMB_RICE_INCREMENT_DIVISOR =              4;

static const Int RExt__PREDICTION_WEIGHTING_ANALYSIS_DC_PRECISION = 0; ///< Additional fixed bit precision used during encoder-side weighting prediction analysis. Currently only used when high_precision_prediction_weighting_flag is set, for backwards compatibility reasons.

static const Int MAX_TIMECODE_SEI_SETS =                            3; ///< Maximum number of time sets

static const Int MAX_CU_DEPTH =                                     6; ///< log2(CTUSize)
static const Int MAX_CU_SIZE =                                     64; ///< = 1<<(MAX_CU_DEPTH)
static const Int MIN_PU_SIZE =                                      4;
static const Int MIN_TU_SIZE =                                      4;
static const Int MAX_TU_SIZE =                                     32;
static const Int MAX_NUM_PART_IDXS_IN_CTU_WIDTH = MAX_CU_SIZE/MIN_PU_SIZE; ///< maximum number of partition indices across the width of a CTU (or height of a CTU)
static const Int SCALING_LIST_REM_NUM =                             6;

static const Int QUANT_SHIFT =                                     14; ///< Q(4) = 2^14
static const Int IQUANT_SHIFT =                                     6;
static const Int SCALE_BITS =                                      15; ///< For fractional bit estimates in RDOQ

static const Int SCALING_LIST_NUM = MAX_NUM_COMPONENT * NUMBER_OF_PREDICTION_MODES; ///< list number for quantization matrix

static const Int SCALING_LIST_START_VALUE =                        8 ; ///< start value for dpcm mode
static const Int MAX_MATRIX_COEF_NUM =                            64 ; ///< max coefficient number for quantization matrix
static const Int MAX_MATRIX_SIZE_NUM =                             8 ; ///< max size number for quantization matrix
static const Int SCALING_LIST_BITS =                               8 ; ///< bit depth of scaling list entries
static const Int LOG2_SCALING_LIST_NEUTRAL_VALUE =                 4 ; ///< log2 of the value that, when used in a scaling list, has no effect on quantisation
static const Int SCALING_LIST_DC =                                16 ; ///< default DC value

static const Int CONTEXT_STATE_BITS =                              6 ;
static const Int LAST_SIGNIFICANT_GROUPS =                        10 ;

static const Int MAX_ENCODER_DEBLOCKING_QUALITY_LAYERS =           8 ;

static const UInt LUMA_LEVEL_TO_DQP_LUT_MAXSIZE =                1024; ///< max LUT size for QP offset based on luma

// ====================================================================================================================
// Macro functions
// ====================================================================================================================

template <typename T> inline T Clip3 (const T minVal, const T maxVal, const T a) { return std::min<T> (std::max<T> (minVal, a) , maxVal); }  ///< general min/max clip
template <typename T> inline T ClipBD(const T x, const Int bitDepth)             { return Clip3(T(0), T((1 << bitDepth)-1), x);           }

template <typename T> inline Void Check3( T minVal, T maxVal, T a)
{
  if ((a > maxVal) || (a < minVal))
  {
    std::cerr << "ERROR: Range check " << minVal << " >= " << a << " <= " << maxVal << " failed" << std::endl;
    assert(false);
    exit(1);
  }
}  ///< general min/max clip

#define DATA_ALIGN                  1                                                                 ///< use 32-bit aligned malloc/free
#if     DATA_ALIGN && _WIN32 && ( _MSC_VER > 1300 )
#define xMalloc( type, len )        _aligned_malloc( sizeof(type)*(len), 32 )
#define xFree( ptr )                _aligned_free  ( ptr )
#else
#define xMalloc( type, len )        malloc   ( sizeof(type)*(len) )
#define xFree( ptr )                free     ( ptr )
#endif

#define FATAL_ERROR_0(MESSAGE, EXITCODE)                      \
{                                                             \
  printf(MESSAGE);                                            \
  exit(EXITCODE);                                             \
}

template <typename ValueType> inline ValueType leftShift       (const ValueType value, const Int shift) { return (shift >= 0) ? ( value                                  << shift) : ( value                                   >> -shift); }
template <typename ValueType> inline ValueType rightShift      (const ValueType value, const Int shift) { return (shift >= 0) ? ( value                                  >> shift) : ( value                                   << -shift); }
template <typename ValueType> inline ValueType leftShift_round (const ValueType value, const Int shift) { return (shift >= 0) ? ( value                                  << shift) : ((value + (ValueType(1) << (-shift - 1))) >> -shift); }
template <typename ValueType> inline ValueType rightShift_round(const ValueType value, const Int shift) { return (shift >= 0) ? ((value + (ValueType(1) << (shift - 1))) >> shift) : ( value                                   << -shift); }
#if O0043_BEST_EFFORT_DECODING
// when shift = 0, returns value
// when shift = 1, (value + 0 + value[1]) >> 1
// when shift = 2, (value + 1 + value[2]) >> 2
// when shift = 3, (value + 3 + value[3]) >> 3
template <typename ValueType> inline ValueType rightShiftEvenRounding(const ValueType value, const UInt shift) { return (shift == 0) ? value : ((value + (1<<(shift-1))-1 + ((value>>shift)&1)) >> shift) ; }
#endif

//! \}

#endif // end of #ifndef  __COMMONDEF__

