/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2025, ITU/ISO/IEC
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

/** \file     BitStream.cpp
    \brief    class for handling bitstream
*/

#include <stdint.h>
#include <vector>
#include "BitStream.h"
#include <string.h>
#include <memory.h>

//! \ingroup CommonLib
//! \{

// ====================================================================================================================
// Constructor / destructor / create / destroy
// ====================================================================================================================

OutputBitstream::OutputBitstream()
{
  clear();
}

OutputBitstream::~OutputBitstream()
{
}

InputBitstream::InputBitstream()
  : m_fifo(), m_emulationPreventionByteLocation(), m_fifoIdx(0), m_numHeldBits(0), m_heldBits(0), m_numBitsRead(0)
{ }

InputBitstream::InputBitstream(const InputBitstream &src)
  : m_fifo(src.m_fifo)
  , m_emulationPreventionByteLocation(src.m_emulationPreventionByteLocation)
  , m_fifoIdx(src.m_fifoIdx)
  , m_numHeldBits(src.m_numHeldBits)
  , m_heldBits(src.m_heldBits)
  , m_numBitsRead(src.m_numBitsRead)
{ }

// ====================================================================================================================
// Public member functions
// ====================================================================================================================

void InputBitstream::resetToStart()
{
  m_fifoIdx     = 0;
  m_numHeldBits = 0;
  m_heldBits    = 0;
  m_numBitsRead = 0;
}

const uint8_t *OutputBitstream::getByteStream() const { return m_fifo.data(); }

uint32_t OutputBitstream::getByteStreamLength()
{
  return uint32_t(m_fifo.size());
}

void OutputBitstream::clear()
{
  m_fifo.clear();
  m_heldBits    = 0;
  m_numHeldBits = 0;
}

void OutputBitstream::write(uint32_t bits, uint32_t numberOfBits)
{
  /* any modulo 8 remainder of numTotalBits cannot be written this time,
   * and will be held until next time. */
  uint32_t numTotalBits    = numberOfBits + m_numHeldBits;
  uint32_t nextNumHeldBits = numTotalBits % BITS_PER_BYTE;

  /* form a byte aligned word (writeBits), by concatenating any held bits
   * with the new bits, discarding the bits that will form the nextHeldBits.
   * eg: H = held bits, V = n new bits        /---- nextHeldBits
   * len(H)=7, len(V)=1: ... ---- HHHH HHHV . 0000 0000, nextNumHeldBits=0
   * len(H)=7, len(V)=2: ... ---- HHHH HHHV . V000 0000, nextNumHeldBits=1
   * if total_bits < 8, the value of v_ is not used */
  uint8_t nextHeldBits = bits << (BITS_PER_BYTE - nextNumHeldBits);

  if (numTotalBits < BITS_PER_BYTE)
  {
    /* insufficient bits accumulated to write out, append new_held_bits to
     * current held_bits */
    /* NB, this requires that v only contains 0 in bit positions {31..n} */
    m_heldBits |= nextHeldBits;
    m_numHeldBits = nextNumHeldBits;
    return;
  }

  /* topword serves to justify held_bits to align with the msb of bits */
  uint32_t topword   = (numberOfBits - nextNumHeldBits) & ~BITS_PER_BYTE_MASK;
  uint32_t writeBits = (m_heldBits << topword) | (bits >> nextNumHeldBits);

  switch (numTotalBits >> BITS_PER_BYTE_LOG2)
  {
  case 4:
    m_fifo.push_back(writeBits >> 3 * BITS_PER_BYTE);
  case 3:
    m_fifo.push_back(writeBits >> 2 * BITS_PER_BYTE);
  case 2:
    m_fifo.push_back(writeBits >> BITS_PER_BYTE);
  case 1:
    m_fifo.push_back(writeBits);
  }

  m_heldBits    = nextHeldBits;
  m_numHeldBits = nextNumHeldBits;
}

void OutputBitstream::writeAlignOne()
{
  const uint32_t numBits = getNumBitsUntilByteAligned();
  write((1 << numBits) - 1, numBits);
  return;
}

void OutputBitstream::writeAlignZero()
{
  if (0 == m_numHeldBits)
  {
    return;
  }
  m_fifo.push_back(m_heldBits);
  m_heldBits    = 0;
  m_numHeldBits = 0;
}

/**
 - add substream to the end of the current bitstream
 .
 \param  pcSubstream  substream to be added
 */
void   OutputBitstream::addSubstream( OutputBitstream* pcSubstream )
{
  uint32_t numBits = pcSubstream->getNumberOfWrittenBits();

  const std::vector<uint8_t> &rbsp = pcSubstream->getFifo();
  for (const uint8_t byte: rbsp)
  {
    write(byte, BITS_PER_BYTE);
  }

  const uint32_t numTrailingBits = numBits & BITS_PER_BYTE_MASK;

  if (numTrailingBits != 0)
  {
    write(pcSubstream->getHeldBits() >> (BITS_PER_BYTE - numTrailingBits), numTrailingBits);
  }
}

void OutputBitstream::writeByteAlignment()
{
  write(1, 1);
  writeAlignZero();
}

int OutputBitstream::countStartCodeEmulations()
{
  uint32_t cnt = 0;
  std::vector<uint8_t> &rbsp = getFifo();
  for (std::vector<uint8_t>::iterator it = rbsp.begin(); it != rbsp.end();)
  {
    std::vector<uint8_t>::iterator found = it;
    do
    {
      // find the next emulated 00 00 {00,01,02,03}
      // NB, end()-1, prevents finding a trailing two byte sequence
      found = search_n(found, rbsp.end()-1, 2, 0);
      found++;
      // if not found, found == end, otherwise found = second zero byte
      if (found == rbsp.end())
      {
        break;
      }
      if (*(++found) <= 3)
      {
        break;
      }
    } while (true);
    it = found;
    if (found != rbsp.end())
    {
      cnt++;
    }
  }
  return cnt;
}

/**
 * read numberOfBits from bitstream without updating the bitstream
 * state, storing the result in bits.
 *
 * If reading numberOfBits would overrun the bitstream buffer,
 * the bitstream is effectively padded with sufficient zero-bits to
 * avoid the overrun.
 */
void InputBitstream::pseudoRead(uint32_t numberOfBits, uint32_t &bits)
{
  uint32_t savedNumHeldBits = m_numHeldBits;
  uint8_t  savedHeldBits    = m_heldBits;
  uint32_t savedFifoIdx     = m_fifoIdx;

  uint32_t numBitsToRead = std::min(numberOfBits, getNumBitsLeft());
  read(numBitsToRead, bits);
  bits <<= (numberOfBits - numBitsToRead);

  m_fifoIdx     = savedFifoIdx;
  m_heldBits    = savedHeldBits;
  m_numHeldBits = savedNumHeldBits;
}

void InputBitstream::read(uint32_t numberOfBits, uint32_t &ruiBits)
{
  m_numBitsRead += numberOfBits;

  /* NB, bits are extracted from the MSB of each byte. */
  uint32_t retval = 0;
  if (numberOfBits <= m_numHeldBits)
  {
    /* n=1, len(H)=7:   -VHH HHHH, shift_down=6, mask=0xfe
     * n=3, len(H)=7:   -VVV HHHH, shift_down=4, mask=0xf8
     */
    retval = m_heldBits >> (m_numHeldBits - numberOfBits);
    retval &= ~(BYTE_MASK << numberOfBits);
    m_numHeldBits -= numberOfBits;
    ruiBits = retval;
    return;
  }

  /* all num_held_bits will go into retval
   *   => need to mask leftover bits from previous extractions
   *   => align retval with top of extracted word */
  /* n=5, len(H)=3: ---- -VVV, mask=0x07, shift_up=5-3=2,
   * n=9, len(H)=3: ---- -VVV, mask=0x07, shift_up=9-3=6 */
  numberOfBits -= m_numHeldBits;
  retval = m_heldBits & ~(BYTE_MASK << m_numHeldBits);
  retval <<= numberOfBits;

  /* number of whole bytes that need to be loaded to form retval */
  /* n=32, len(H)=0, load 4bytes, shift_down=0
   * n=32, len(H)=1, load 4bytes, shift_down=1
   * n=31, len(H)=1, load 4bytes, shift_down=1+1
   * n=8,  len(H)=0, load 1byte,  shift_down=0
   * n=8,  len(H)=3, load 1byte,  shift_down=3
   * n=5,  len(H)=1, load 1byte,  shift_down=1+3
   */
  uint32_t alignedWord       = 0;
  uint32_t num_bytes_to_load = (numberOfBits - 1) >> BITS_PER_BYTE_LOG2;

  switch (num_bytes_to_load)
  {
  case 3:
    alignedWord = m_fifo[m_fifoIdx++] << 3 * BITS_PER_BYTE;
  case 2:
    alignedWord |= m_fifo[m_fifoIdx++] << 2 * BITS_PER_BYTE;
  case 1:
    alignedWord |= m_fifo[m_fifoIdx++] << BITS_PER_BYTE;
  case 0:
    alignedWord |= m_fifo[m_fifoIdx++];
  }

  /* resolve remainder bits */
  uint32_t nextNumHeldBits = (BITS_PER_WORD - numberOfBits) % BITS_PER_BYTE;

  /* copy required part of alignedWord into retval */
  retval |= alignedWord >> nextNumHeldBits;

  /* store held bits */
  m_numHeldBits = nextNumHeldBits;
  m_heldBits    = alignedWord;

  ruiBits = retval;
}

/**
 * insert the contents of the bytealigned (and flushed) bitstream src
 * into this at byte position pos.
 */
void OutputBitstream::insertAt(const OutputBitstream& src, uint32_t pos)
{
  m_fifo.insert(m_fifo.begin() + pos, src.m_fifo.begin(), src.m_fifo.end());
}

uint32_t InputBitstream::readOutTrailingBits ()
{
  uint32_t count = 0;
  uint32_t bits  = 0;

  while (getNumBitsLeft() > 0 && getNumBitsUntilByteAligned() != 0)
  {
    count++;
    read(1, bits);
  }
  return count;
}

uint32_t InputBitstream::readByteAlignment()
{
  uint32_t code = 0;
  read( 1, code );

  const uint32_t numBits = getNumBitsUntilByteAligned();
  if (numBits > 0)
  {
    read( numBits, code );
  }
  return numBits + 1;
}

//! \}
