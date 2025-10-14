/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#include "common_randomness.h"
#include <math.h>

// Common randomness noise generator
// Identical to the one used in Python: integer random generator shapes into a Normal distribution
float common_randomness::grand()
{
    unsigned long int  a = 16807;
    unsigned long int  m = 2147483647;
    double          pi = 3.14159265359;
    double          u1,u2;

    this->seed = ( a * this->seed ) % m;
    u1 = (double)this->seed / (double)m;

    this->seed = ( a * this->seed ) % m;
    u2= (double)this->seed / (double)m;

//     return 0;
    return sqrt( -2. * log( u1 )) * cos( 2. * pi * u2 );
}

short common_randomness::grand16()
{
    return (short)( 256. * grand() );
}
