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

/** \file     TEncBinCoderCABAC.h
    \brief    binary entropy encoder of CABAC
*/

#ifndef __TENCBINCODERCABAC__
#define __TENCBINCODERCABAC__

#include "TEncBinCoder.h"

#ifdef ENTROPY_CODING_DEBUG
extern int g_epbits;
extern int g_trmbits;
#endif

//! \ingroup TLibEncoder
//! \{

class TEncBinCABAC : public TEncBinIf
{
public:
  TEncBinCABAC ();
  virtual ~TEncBinCABAC();

  Void  init              ( OutputBitstream* pcTComBitstream );
  Void  uninit            ();

  Void  start             ();
  Void  finish            ();

  unsigned  getNumWrittenBits()
  {
    return (m_bitstream->getNumberOfWrittenBits() + 8 * m_numBufferedBytes + 23 - m_bitsLeft);
  }


  Void  encodeBin         ( BinProbModel_Std &probModel, unsigned bin, bool d_update = false );
  Void  encodeBinsEP      ( unsigned bins, unsigned numBins );
  Void  encodeBinEP       ( unsigned bin );
  Void  encodeExGolomb    ( unsigned val, unsigned count );
  Void  encodeBinTrm      ( unsigned  bin );

  Void  align             ();
  Void  encodeAlignedBinsEP( unsigned  binValues, unsigned numBins             );

private:
  Void writeOut();

  OutputBitstream        *m_bitstream;
  uint32_t                m_low;
  uint32_t                m_range;
  uint32_t                m_bufferedByte;
  int32_t                 m_numBufferedBytes;
  int32_t                 m_bitsLeft;
};

//! \}

#endif

