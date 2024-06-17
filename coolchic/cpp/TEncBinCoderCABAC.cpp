/*
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

/** \file     TEncBinCoderCABAC.cpp
    \brief    binary entropy encoder of CABAC
*/

#include "TEncBinCoderCABAC.h"

#ifdef ENTROPY_CODING_DEBUG
int g_epbits;
int g_trmbits;
#endif

//! \ingroup TLibEncoder
//! \{


TEncBinCABAC::TEncBinCABAC()
: m_bitstream( 0 )
{
}

TEncBinCABAC::~TEncBinCABAC()
{
}

Void TEncBinCABAC::init( OutputBitstream* pcTComBitstream )
{
  m_bitstream = pcTComBitstream;
}

Void TEncBinCABAC::uninit()
{
  m_bitstream = 0;
}

Void TEncBinCABAC::start()
{
  m_low               = 0;
  m_range             = 510;
  m_bufferedByte      = 0xff;
  m_numBufferedBytes  = 0;
  m_bitsLeft          = 23;
}

Void TEncBinCABAC::finish()
{
#ifdef ENTROPY_CODING_DEBUG
  int enc_start = getNumWrittenBits();
#endif
  if (m_low >> (32 - m_bitsLeft))
  {
    m_bitstream->write(m_bufferedByte + 1, 8);
    while( m_numBufferedBytes > 1 )
    {
      m_bitstream->write(0x00, 8);
      m_numBufferedBytes--;
    }
    m_low -= 1 << (32 - m_bitsLeft);
  }
  else
  {
    if( m_numBufferedBytes > 0 )
    {
      m_bitstream->write(m_bufferedByte, 8);
    }
    while( m_numBufferedBytes > 1 )
    {
      m_bitstream->write(0xff, 8);
      m_numBufferedBytes--;
    }
  }
  m_bitstream->write(m_low >> 8, 24 - m_bitsLeft);
#ifdef ENTROPY_CODING_DEBUG
  int enc_end = getNumWrittenBits();
  printf("finishbits=%d\n", enc_end-enc_start);
#endif
}

/**
 * \brief Encode bin
 *
 * \param binValue   bin value
 * \param rcCtxModel context model
 */
Void TEncBinCABAC::encodeBin( BinProbModel_Std &probModel, unsigned bin, bool do_update )
{
  uint32_t      lpsRange  = probModel.getLPS(m_range);

#ifdef ENTROPY_CODING_DEBUG
  int enc_start = (int)getNumWrittenBits();
#endif
  m_range -= lpsRange;
  if (bin != probModel.mps())
  {
    int numBits = probModel.getRenormBitsLPS(lpsRange);
    m_bitsLeft   -= numBits;
    m_low += m_range;
    m_low   = m_low << numBits;
    m_range = lpsRange << numBits;
    if( m_bitsLeft < 12 )
    {
      writeOut();
    }
  }
  else
  {
    if (m_range < 256)
    {
      int numBits = probModel.getRenormBitsRange(m_range);
      m_bitsLeft   -= numBits;
      m_low <<= numBits;
      m_range <<= numBits;
      if( m_bitsLeft < 12 )
      {
        writeOut();
      }
    }
  }
#ifdef ENTROPY_CODING_DEBUG
  probModel.m_binCnt++;
  int enc_end = (int)getNumWrittenBits();
  probModel.m_numBits += (enc_end-enc_start);
  if (bin == 0)
      probModel.m_zeroCnt++;
#endif
  if (do_update)
      probModel.update(bin);
}

/**
 * \brief Encode equiprobable bin
 *
 * \param binValue bin value
 */
Void TEncBinCABAC::encodeBinEP( unsigned bin )
{
#ifdef ENTROPY_CODING_DEBUG
  int enc_start = (int)getNumWrittenBits();
#endif
  m_low <<= 1;
  if( bin )
  {
    m_low += m_range;
  }
  m_bitsLeft--;
  if( m_bitsLeft < 12 )
  {
    writeOut();
  }
#ifdef ENTROPY_CODING_DEBUG
  int enc_end = (int)getNumWrittenBits();
  g_epbits += enc_end-enc_start;
#endif
}

Void  TEncBinCABAC::encodeExGolomb( unsigned symbol, unsigned count )
{
    int bins = 0;
    int nbins = 0;
    while (symbol >= (unsigned)(1<<count))
    {
        bins = 2*bins+1;
        nbins += 1;
        symbol -= 1<<count;
        count += 1;
    }
    bins = 2*bins+0;
    nbins += 1;
    bins = (bins<<count) | symbol;
    nbins += count;
    if (nbins > 32)
    {
        printf("encodeexgolomb overflow: %d bits\n", nbins);
        exit(1);
    }
    encodeBinsEP(bins, nbins);
}

/**
 * \brief Encode equiprobable bins
 *
 * \param binValues bin values
 * \param numBins number of bins
 */
Void TEncBinCABAC::encodeBinsEP( unsigned bins, unsigned numBins )
{
#ifdef ENTROPY_CODING_DEBUG
  int enc_start = getNumWrittenBits();
#endif
  if (m_range == 256)
  {
    encodeAlignedBinsEP( bins, numBins );
#ifdef ENTROPY_CODING_DEBUG
    int enc_end = getNumWrittenBits();
    g_epbits += enc_end-enc_start;
#endif
    return;
  }
  while( numBins > 8 )
  {
    numBins          -= 8;
    unsigned pattern  = bins >> numBins;
    m_low <<= 8;
    m_low += m_range * pattern;
    bins             -= pattern << numBins;
    m_bitsLeft       -= 8;
    if( m_bitsLeft < 12 )
    {
      writeOut();
    }
  }
  m_low <<= numBins;
  m_low += m_range * bins;
  m_bitsLeft -= numBins;
  if( m_bitsLeft < 12 )
  {
    writeOut();
  }
#ifdef ENTROPY_CODING_DEBUG
  int enc_end = getNumWrittenBits();
  g_epbits += enc_end-enc_start;
#endif
}

void TEncBinCABAC::encodeAlignedBinsEP( unsigned bins, unsigned numBins )
{
  unsigned remBins = numBins;
  while( remBins > 0 )
  {
    //The process of encoding an EP bin is the same as that of coding a normal
    //bin where the symbol ranges for 1 and 0 are both half the range:
    //
    //  low = (low + range/2) << 1       (to encode a 1)
    //  low =  low            << 1       (to encode a 0)
    //
    //  i.e.
    //  low = (low + (bin * range/2)) << 1
    //
    //  which is equivalent to:
    //
    //  low = (low << 1) + (bin * range)
    //
    //  this can be generalised for multiple bins, producing the following expression:
    //
    unsigned binsToCode = std::min<unsigned>( remBins, 8); //code bytes if able to take advantage of the system's byte-write function
    unsigned binMask    = ( 1 << binsToCode ) - 1;
    unsigned newBins    = ( bins >> ( remBins - binsToCode ) ) & binMask;
    m_low               = (m_low << binsToCode) + (newBins << 8);   // range is known to be 256
    remBins            -= binsToCode;
    m_bitsLeft         -= binsToCode;
    if( m_bitsLeft < 12 )
    {
      writeOut();
    }
  }
}

Void TEncBinCABAC::align()
{
  m_range = 256;
}

/**
 * \brief Encode terminating bin
 *
 * \param binValue bin value
 */
Void TEncBinCABAC::encodeBinTrm( unsigned bin )
{
#ifdef ENTROPY_CODING_DEBUG
  int enc_start = (int)getNumWrittenBits();
#endif
  m_range -= 2;
  if( bin )
  {
    m_low += m_range;
    m_low <<= 7;
    m_range = 2 << 7;
    m_bitsLeft -= 7;
  }
  else if (m_range >= 256)
  {
#ifdef ENTROPY_CODING_DEBUG
    int enc_end = (int)getNumWrittenBits();
    printf("trm=%d\n", enc_end-enc_start);
#endif
    return;
  }
  else
  {
    m_low <<= 1;
    m_range <<= 1;
    m_bitsLeft--;
  }
  if( m_bitsLeft < 12 )
  {
    writeOut();
  }
#ifdef ENTROPY_CODING_DEBUG
  int enc_end = (int)getNumWrittenBits();
  printf("trm=%d\n", enc_end-enc_start);
#endif
}

/**
 * \brief Move bits from register into bitstream
 */
Void TEncBinCABAC::writeOut()
{
  unsigned leadByte = m_low >> (24 - m_bitsLeft);
  m_bitsLeft       += 8;
  m_low &= 0xffffffffu >> m_bitsLeft;
  if( leadByte == 0xff )
  {
    m_numBufferedBytes++;
  }
  else
  {
    if( m_numBufferedBytes > 0 )
    {
      unsigned carry  = leadByte >> 8;
      unsigned byte   = m_bufferedByte + carry;
      m_bufferedByte  = leadByte & 0xff;
      m_bitstream->write(byte, 8);
      byte            = ( 0xff + carry ) & 0xff;
      while( m_numBufferedBytes > 1 )
      {
        m_bitstream->write(byte, 8);
        m_numBufferedBytes--;
      }
    }
    else
    {
      m_numBufferedBytes  = 1;
      m_bufferedByte      = leadByte;
    }
  }
}

//! \}
