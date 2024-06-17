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

/** \file     TDecBinCoderCABAC.h
    \brief    binary entropy decoder of CABAC
*/

#ifndef __TDECBINCODERCABAC__
#define __TDECBINCODERCABAC__

#include "TDecBinCoder.h"

//! \ingroup TLibDecoder
//! \{

class TDecBinCABAC : public TDecBinIf
{
public:
  TDecBinCABAC ();
  virtual ~TDecBinCABAC();

  Void  init              ( InputBitstream* pcTComBitstream );
  Void  uninit            ();

  Void  start             ();
  Void  finish            ();

  UInt  decodeBin         ( BinProbModel_Std& probModel, bool do_update = false )
                                {
                                  unsigned bin = probModel.mps();
                                  uint32_t      lpsRange  = probModel.getLPS(m_range);

                                  m_range -= lpsRange; 
                                  uint32_t scaledRange = m_range << 7;
                                  if (m_value < scaledRange)
                                  {
                                    // MPS path
                                    if (m_range < 256)
                                    {
                                      int numBits = probModel.getRenormBitsRange(m_range);
                                      m_range <<= numBits;
                                      m_value <<= numBits;
                                      m_bitsNeeded += numBits;
                                      if( m_bitsNeeded >= 0 )
                                      {
                                        m_value += m_bitstream->readByte() << m_bitsNeeded;
                                        m_bitsNeeded -= 8;
                                      }
                                    }
                                  }
                                  else
                                  {
                                    // LPS path
                                    bin = 1 - bin;
                                    int numBits = probModel.getRenormBitsLPS(lpsRange);
                                    m_value -= scaledRange;
                                    m_value = m_value << numBits;
                                    m_range = lpsRange << numBits;
                                    m_bitsNeeded += numBits;
                                    if( m_bitsNeeded >= 0 )
                                    {
                                      m_value += m_bitstream->readByte() << m_bitsNeeded;
                                      m_bitsNeeded -= 8;
                                    }
                                  }

                                  if (do_update)
                                      probModel.update(bin);

                                  return bin;
                                }
  UInt  decodeBinEP       ( );
  UInt  decodeBinsEP      ( Int numBins              );
  UInt  decodeAlignedBinsEP( Int numBins             );
  Int  decodeExGolomb    ( Int count )
                            {
                              int symbol = 0;
                              UInt bit = 1;
                              while (bit)
                              {
                                bit = decodeBinEP();
                                symbol += bit << count;
                                count += 1;
                              }
                              count -= 1;
                              if (count > 0)
                              {
                                symbol += decodeBinsEP(count);
                              }
                              return symbol;
                            }

  Void  align             ();

  UInt  decodeBinTrm      ( );

  Void  xReadPCMCode      ( UInt uiLength, UInt& ruiCode );

  TDecBinCABAC* getTDecBinCABAC()             { return this; }
  const TDecBinCABAC* getTDecBinCABAC() const { return this; }

  int bitsRead() { return m_bitstream->getNumBitsRead()+m_bitsNeeded; }

private:
  InputBitstream   *m_bitstream;
  uint32_t          m_range;
  uint32_t          m_value;
  int32_t           m_bitsNeeded;
};

//! \}

#endif

