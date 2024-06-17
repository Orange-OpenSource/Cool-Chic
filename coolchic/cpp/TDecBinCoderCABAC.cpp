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

/** \file     TDecBinCoderCABAC.cpp
    \brief    binary entropy decoder of CABAC
*/

#include "TDecBinCoderCABAC.h"

//! \ingroup TLibDecoder
//! \{

TDecBinCABAC::TDecBinCABAC()
: m_bitstream( 0 )
{
}

TDecBinCABAC::~TDecBinCABAC()
{
}

Void
TDecBinCABAC::init( InputBitstream* pcBitstream )
{
  m_bitstream = pcBitstream;
}

Void
TDecBinCABAC::uninit()
{
  m_bitstream = 0;
}

Void
TDecBinCABAC::start()
{
  m_range    = 510;
  m_value    = (m_bitstream->readByte() << 8);
  m_value   |= m_bitstream->readByte();
  m_bitsNeeded = -8;
}

Void
TDecBinCABAC::finish()
{
  UInt lastByte;

  m_bitstream->peekPreviousByte( lastByte );
  // Check for proper stop/alignment pattern
  assert( ((lastByte << (8 + m_bitsNeeded)) & 0xff) == 0x80 );
}

UInt TDecBinCABAC::decodeBinEP( )
{
  m_value += m_value;
  if( ++m_bitsNeeded >= 0 )
  {
    m_value += m_bitstream->readByte();
    m_bitsNeeded      = -8;
  } 
    
  unsigned bin = 0;
  unsigned scaledRange = m_range << 7;
  if (m_value >= scaledRange)
  { 
    m_value -= scaledRange;
    bin        = 1;
  } 
  return bin;
}

UInt TDecBinCABAC::decodeBinsEP( Int numBins )
{

  if (m_range == 256)
  {
    return decodeAlignedBinsEP( numBins );
  }
  unsigned remBins = numBins;
  unsigned bins    = 0;
  while(   remBins > 8 )
  {
    m_value              = (m_value << 8) + (m_bitstream->readByte() << (8 + m_bitsNeeded));
    unsigned scaledRange = m_range << 15;
    for( int i = 0; i < 8; i++ )
    {
      bins += bins;
      scaledRange >>= 1;
      if (m_value >= scaledRange)
      {
        bins    ++;
        m_value -= scaledRange;
      }
    }
    remBins -= 8;
  }
  m_bitsNeeded   += remBins;
  m_value <<= remBins;
  if( m_bitsNeeded >= 0 )
  {
    m_value += m_bitstream->readByte() << m_bitsNeeded;
    m_bitsNeeded -= 8;
  }
  unsigned scaledRange = m_range << (remBins + 7);
  for ( int i = 0; i < (int)remBins; i++ )
  {
    bins += bins;
    scaledRange >>= 1;
    if (m_value >= scaledRange)
    {
      bins    ++;
      m_value -= scaledRange;
    }
  }
  return bins;
}

Void TDecBinCABAC::align()
{
  m_range = 256;
}

UInt TDecBinCABAC::decodeAlignedBinsEP( Int numBins )
{
  unsigned remBins = numBins;
  unsigned bins    = 0;
  while(   remBins > 0 )
  {
    // The MSB of m_value is known to be 0 because range is 256. Therefore:
    //   > The comparison against the symbol range of 128 is simply a test on the next-most-significant bit
    //   > "Subtracting" the symbol range if the decoded bin is 1 simply involves clearing that bit.
    //  As a result, the required bins are simply the <binsToRead> next-most-significant bits of m_value
    //  (m_value is stored MSB-aligned in a 16-bit buffer - hence the shift of 15)
    //
    //    m_value = |0|V|V|V|V|V|V|V|V|B|B|B|B|B|B|B|
    //    (V = usable bit, B = potential buffered bit (buffer refills when m_bitsNeeded >= 0))
    //
    unsigned binsToRead = std::min<unsigned>( remBins, 8 ); //read bytes if able to take advantage of the system's byte-read function
    unsigned binMask    = ( 1 << binsToRead ) - 1;
    unsigned newBins    = (m_value >> (15 - binsToRead)) & binMask;
    bins                = ( bins    << binsToRead) | newBins;
    m_value             = (m_value << binsToRead) & 0x7FFF;
    remBins            -= binsToRead;
    m_bitsNeeded       += binsToRead;
    if( m_bitsNeeded >= 0 )
    {
      m_value |= m_bitstream->readByte() << m_bitsNeeded;
      m_bitsNeeded     -= 8;
    }
  }
  return bins;
}

UInt
TDecBinCABAC::decodeBinTrm( )
{
  m_range -= 2;
  unsigned scaledRange = m_range << 7;
  if (m_value >= scaledRange)
  {
    return 1;
  }
  else
  {
    if (m_range < 256)
    {
      m_range += m_range;
      m_value += m_value;
      if( ++m_bitsNeeded == 0 )
      {
        m_value += m_bitstream->readByte();
        m_bitsNeeded  = -8;
      }
    }
    return 0;
  }
}

/** Read a PCM code.
 * \param uiLength code bit-depth
 * \param ruiCode pointer to PCM code value
 * \returns Void
 */
Void
TDecBinCABAC::xReadPCMCode(UInt uiLength, UInt& ruiCode)
{
  assert ( uiLength > 0 );
  m_bitstream->read (uiLength, ruiCode);
}

//! \}
