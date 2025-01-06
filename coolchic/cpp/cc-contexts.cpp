/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#include "Contexts.h"
#include "common.h"
#include "cc-contexts.h"

MuSigGTs g_contexts[N_MUQ+1][N_SIGQ] = {
{
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,65,65,127 ),
  MuSigGTs( 65,1,1,65,127 ),
  MuSigGTs( 65,1,1,65,127 ),
  MuSigGTs( 65,1,1,65,127 ),
  MuSigGTs( 65,5,5,65,125 ),
  MuSigGTs( 67,9,9,9,121 ),
  MuSigGTs( 69,13,13,13,115 ),
  MuSigGTs( 73,21,21,21,111 ),
  MuSigGTs( 75,29,29,29,105 ),
  MuSigGTs( 81,39,39,39,99 ),
  MuSigGTs( 87,49,49,49,93 ),
  MuSigGTs( 91,59,59,59,89 ),
  MuSigGTs( 95,65,65,65,83 ),
  MuSigGTs( 101,73,73,73,79 ),
  MuSigGTs( 105,81,81,81,75 ),
  MuSigGTs( 109,87,87,87,73 ),
  MuSigGTs( 113,95,95,95,73 ),
  MuSigGTs( 115,99,99,99,71 ),
  MuSigGTs( 117,105,105,105,69 ),
  MuSigGTs( 119,109,109,109,67 ),
  MuSigGTs( 121,113,113,113,67 ),
  MuSigGTs( 123,115,115,115,67 ),
  MuSigGTs( 123,117,117,117,67 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 3,1,65,65,127 ),
  MuSigGTs( 7,1,65,65,127 ),
  MuSigGTs( 9,1,65,65,127 ),
  MuSigGTs( 13,1,65,65,127 ),
  MuSigGTs( 19,1,65,65,127 ),
  MuSigGTs( 23,1,65,65,127 ),
  MuSigGTs( 29,1,65,65,127 ),
  MuSigGTs( 33,1,65,65,127 ),
  MuSigGTs( 37,1,65,65,127 ),
  MuSigGTs( 41,1,65,65,127 ),
  MuSigGTs( 45,1,65,65,127 ),
  MuSigGTs( 49,1,1,65,127 ),
  MuSigGTs( 53,1,1,65,125 ),
  MuSigGTs( 57,5,5,65,123 ),
  MuSigGTs( 61,9,9,9,117 ),
  MuSigGTs( 65,13,13,13,113 ),
  MuSigGTs( 67,21,21,21,107 ),
  MuSigGTs( 73,29,29,29,101 ),
  MuSigGTs( 79,39,39,39,95 ),
  MuSigGTs( 83,49,49,49,89 ),
  MuSigGTs( 89,59,59,59,85 ),
  MuSigGTs( 95,65,65,65,81 ),
  MuSigGTs( 99,73,73,73,79 ),
  MuSigGTs( 105,81,81,81,75 ),
  MuSigGTs( 107,87,87,87,73 ),
  MuSigGTs( 111,95,95,95,71 ),
  MuSigGTs( 115,99,99,99,69 ),
  MuSigGTs( 117,105,105,105,67 ),
  MuSigGTs( 119,109,109,109,67 ),
  MuSigGTs( 121,113,113,113,67 ),
  MuSigGTs( 123,115,115,115,67 ),
  MuSigGTs( 123,117,117,117,65 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 3,1,65,65,127 ),
  MuSigGTs( 5,1,65,65,127 ),
  MuSigGTs( 9,1,65,65,127 ),
  MuSigGTs( 11,1,65,65,127 ),
  MuSigGTs( 17,1,65,65,127 ),
  MuSigGTs( 21,1,65,65,127 ),
  MuSigGTs( 27,1,65,65,127 ),
  MuSigGTs( 31,1,65,65,127 ),
  MuSigGTs( 37,1,1,65,125 ),
  MuSigGTs( 41,1,1,65,123 ),
  MuSigGTs( 47,5,5,65,119 ),
  MuSigGTs( 53,9,9,9,113 ),
  MuSigGTs( 59,13,13,13,107 ),
  MuSigGTs( 65,21,21,21,101 ),
  MuSigGTs( 69,29,29,29,95 ),
  MuSigGTs( 75,39,39,39,91 ),
  MuSigGTs( 81,49,49,49,87 ),
  MuSigGTs( 89,59,59,59,81 ),
  MuSigGTs( 93,65,65,65,79 ),
  MuSigGTs( 99,73,73,73,75 ),
  MuSigGTs( 103,81,81,81,73 ),
  MuSigGTs( 107,87,87,87,71 ),
  MuSigGTs( 111,95,95,95,71 ),
  MuSigGTs( 115,99,99,99,69 ),
  MuSigGTs( 117,105,105,105,67 ),
  MuSigGTs( 119,109,109,109,67 ),
  MuSigGTs( 121,113,113,113,67 ),
  MuSigGTs( 123,115,115,115,67 ),
  MuSigGTs( 123,117,117,117,65 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 3,1,65,65,127 ),
  MuSigGTs( 5,1,65,65,127 ),
  MuSigGTs( 7,1,65,65,127 ),
  MuSigGTs( 11,1,65,65,127 ),
  MuSigGTs( 17,1,65,65,127 ),
  MuSigGTs( 21,1,65,65,127 ),
  MuSigGTs( 27,1,1,65,123 ),
  MuSigGTs( 33,1,1,65,119 ),
  MuSigGTs( 39,5,5,65,115 ),
  MuSigGTs( 47,9,9,9,109 ),
  MuSigGTs( 55,13,13,13,103 ),
  MuSigGTs( 63,21,21,21,95 ),
  MuSigGTs( 67,29,29,29,91 ),
  MuSigGTs( 73,39,39,39,87 ),
  MuSigGTs( 79,49,49,49,81 ),
  MuSigGTs( 87,59,59,59,79 ),
  MuSigGTs( 93,65,65,65,75 ),
  MuSigGTs( 99,73,73,73,73 ),
  MuSigGTs( 103,81,81,81,71 ),
  MuSigGTs( 107,87,87,87,71 ),
  MuSigGTs( 111,95,95,95,69 ),
  MuSigGTs( 115,99,99,99,67 ),
  MuSigGTs( 117,105,105,105,67 ),
  MuSigGTs( 119,109,109,109,67 ),
  MuSigGTs( 121,113,113,113,67 ),
  MuSigGTs( 123,115,115,115,65 ),
  MuSigGTs( 123,117,117,117,65 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 3,1,65,65,127 ),
  MuSigGTs( 7,1,65,65,127 ),
  MuSigGTs( 11,1,65,65,125 ),
  MuSigGTs( 15,1,65,65,123 ),
  MuSigGTs( 21,1,1,65,119 ),
  MuSigGTs( 27,1,1,65,113 ),
  MuSigGTs( 33,5,5,65,107 ),
  MuSigGTs( 41,9,9,9,101 ),
  MuSigGTs( 51,13,13,13,95 ),
  MuSigGTs( 59,21,21,21,91 ),
  MuSigGTs( 65,29,29,29,87 ),
  MuSigGTs( 71,39,39,39,81 ),
  MuSigGTs( 79,49,49,49,79 ),
  MuSigGTs( 87,59,59,59,75 ),
  MuSigGTs( 93,65,65,65,73 ),
  MuSigGTs( 99,73,73,73,71 ),
  MuSigGTs( 103,81,81,81,71 ),
  MuSigGTs( 107,87,87,87,69 ),
  MuSigGTs( 111,95,95,95,67 ),
  MuSigGTs( 113,99,99,99,67 ),
  MuSigGTs( 117,105,105,105,67 ),
  MuSigGTs( 119,109,109,109,65 ),
  MuSigGTs( 121,113,113,113,65 ),
  MuSigGTs( 121,115,115,115,65 ),
  MuSigGTs( 123,117,117,117,65 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 3,1,65,65,125 ),
  MuSigGTs( 7,1,65,65,121 ),
  MuSigGTs( 11,1,65,65,117 ),
  MuSigGTs( 17,1,65,65,111 ),
  MuSigGTs( 23,1,1,65,105 ),
  MuSigGTs( 29,5,5,65,99 ),
  MuSigGTs( 39,9,9,9,93 ),
  MuSigGTs( 47,13,13,13,89 ),
  MuSigGTs( 57,21,21,21,83 ),
  MuSigGTs( 65,29,29,29,79 ),
  MuSigGTs( 71,39,39,39,75 ),
  MuSigGTs( 79,49,49,49,75 ),
  MuSigGTs( 85,59,59,59,73 ),
  MuSigGTs( 91,65,65,65,71 ),
  MuSigGTs( 99,73,73,73,69 ),
  MuSigGTs( 103,81,81,81,67 ),
  MuSigGTs( 107,87,87,87,67 ),
  MuSigGTs( 111,95,95,95,67 ),
  MuSigGTs( 113,99,99,99,67 ),
  MuSigGTs( 117,105,105,105,65 ),
  MuSigGTs( 119,109,109,109,65 ),
  MuSigGTs( 121,113,113,113,65 ),
  MuSigGTs( 121,115,115,115,65 ),
  MuSigGTs( 123,117,117,117,65 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 3,1,65,65,117 ),
  MuSigGTs( 5,1,65,65,111 ),
  MuSigGTs( 7,1,65,65,105 ),
  MuSigGTs( 13,1,65,65,99 ),
  MuSigGTs( 19,1,1,65,93 ),
  MuSigGTs( 27,5,5,65,89 ),
  MuSigGTs( 35,9,9,9,83 ),
  MuSigGTs( 45,13,13,13,79 ),
  MuSigGTs( 55,21,21,21,75 ),
  MuSigGTs( 63,29,29,29,75 ),
  MuSigGTs( 69,39,39,39,73 ),
  MuSigGTs( 75,49,49,49,71 ),
  MuSigGTs( 85,59,59,59,69 ),
  MuSigGTs( 91,65,65,65,67 ),
  MuSigGTs( 95,73,73,73,67 ),
  MuSigGTs( 103,81,81,81,67 ),
  MuSigGTs( 107,87,87,87,67 ),
  MuSigGTs( 111,95,95,95,65 ),
  MuSigGTs( 113,99,99,99,65 ),
  MuSigGTs( 115,105,105,105,65 ),
  MuSigGTs( 119,109,109,109,65 ),
  MuSigGTs( 121,113,113,113,65 ),
  MuSigGTs( 121,115,115,115,65 ),
  MuSigGTs( 123,117,117,117,65 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,127 ),
  MuSigGTs( 1,1,65,65,103 ),
  MuSigGTs( 1,1,65,65,95 ),
  MuSigGTs( 3,1,65,65,91 ),
  MuSigGTs( 7,1,65,65,87 ),
  MuSigGTs( 11,1,65,65,81 ),
  MuSigGTs( 17,1,1,65,79 ),
  MuSigGTs( 25,5,5,65,75 ),
  MuSigGTs( 33,9,9,9,73 ),
  MuSigGTs( 43,13,13,13,71 ),
  MuSigGTs( 53,21,21,21,71 ),
  MuSigGTs( 63,29,29,29,69 ),
  MuSigGTs( 69,39,39,39,67 ),
  MuSigGTs( 75,49,49,49,67 ),
  MuSigGTs( 83,59,59,59,67 ),
  MuSigGTs( 91,65,65,65,67 ),
  MuSigGTs( 95,73,73,73,65 ),
  MuSigGTs( 103,81,81,81,65 ),
  MuSigGTs( 107,87,87,87,65 ),
  MuSigGTs( 111,95,95,95,65 ),
  MuSigGTs( 113,99,99,99,65 ),
  MuSigGTs( 115,105,105,105,65 ),
  MuSigGTs( 119,109,109,109,65 ),
  MuSigGTs( 121,113,113,113,65 ),
  MuSigGTs( 121,115,115,115,65 ),
  MuSigGTs( 123,117,117,117,65 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,65 ),
  MuSigGTs( 1,1,65,65,65 ),
  MuSigGTs( 1,1,65,65,65 ),
  MuSigGTs( 3,1,65,65,65 ),
  MuSigGTs( 5,1,65,65,65 ),
  MuSigGTs( 11,1,65,65,65 ),
  MuSigGTs( 17,1,1,65,65 ),
  MuSigGTs( 25,5,5,65,65 ),
  MuSigGTs( 33,9,9,9,65 ),
  MuSigGTs( 43,13,13,13,65 ),
  MuSigGTs( 53,21,21,21,65 ),
  MuSigGTs( 63,29,29,29,65 ),
  MuSigGTs( 67,39,39,39,65 ),
  MuSigGTs( 75,49,49,49,65 ),
  MuSigGTs( 83,59,59,59,65 ),
  MuSigGTs( 91,65,65,65,65 ),
  MuSigGTs( 95,73,73,73,65 ),
  MuSigGTs( 103,81,81,81,65 ),
  MuSigGTs( 107,87,87,87,65 ),
  MuSigGTs( 111,95,95,95,65 ),
  MuSigGTs( 113,99,99,99,65 ),
  MuSigGTs( 115,105,105,105,65 ),
  MuSigGTs( 119,109,109,109,65 ),
  MuSigGTs( 121,113,113,113,65 ),
  MuSigGTs( 121,115,115,115,65 ),
  MuSigGTs( 123,117,117,117,65 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,27 ),
  MuSigGTs( 1,1,65,65,31 ),
  MuSigGTs( 3,1,65,65,37 ),
  MuSigGTs( 7,1,65,65,41 ),
  MuSigGTs( 11,1,65,65,47 ),
  MuSigGTs( 17,1,1,65,49 ),
  MuSigGTs( 25,5,5,65,53 ),
  MuSigGTs( 33,9,9,9,55 ),
  MuSigGTs( 43,13,13,13,57 ),
  MuSigGTs( 53,21,21,21,59 ),
  MuSigGTs( 63,29,29,29,61 ),
  MuSigGTs( 69,39,39,39,61 ),
  MuSigGTs( 75,49,49,49,63 ),
  MuSigGTs( 83,59,59,59,63 ),
  MuSigGTs( 91,65,65,65,63 ),
  MuSigGTs( 95,73,73,73,63 ),
  MuSigGTs( 103,81,81,81,63 ),
  MuSigGTs( 107,87,87,87,65 ),
  MuSigGTs( 111,95,95,95,65 ),
  MuSigGTs( 113,99,99,99,65 ),
  MuSigGTs( 115,105,105,105,65 ),
  MuSigGTs( 119,109,109,109,65 ),
  MuSigGTs( 121,113,113,113,65 ),
  MuSigGTs( 121,115,115,115,65 ),
  MuSigGTs( 123,117,117,117,65 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 3,1,65,65,13 ),
  MuSigGTs( 5,1,65,65,17 ),
  MuSigGTs( 7,1,65,65,23 ),
  MuSigGTs( 13,1,65,65,29 ),
  MuSigGTs( 19,1,1,65,35 ),
  MuSigGTs( 27,5,5,65,41 ),
  MuSigGTs( 35,9,9,9,45 ),
  MuSigGTs( 45,13,13,13,49 ),
  MuSigGTs( 55,21,21,21,51 ),
  MuSigGTs( 63,29,29,29,55 ),
  MuSigGTs( 69,39,39,39,57 ),
  MuSigGTs( 75,49,49,49,57 ),
  MuSigGTs( 85,59,59,59,59 ),
  MuSigGTs( 91,65,65,65,61 ),
  MuSigGTs( 95,73,73,73,61 ),
  MuSigGTs( 103,81,81,81,63 ),
  MuSigGTs( 107,87,87,87,63 ),
  MuSigGTs( 111,95,95,95,63 ),
  MuSigGTs( 113,99,99,99,63 ),
  MuSigGTs( 115,105,105,105,63 ),
  MuSigGTs( 119,109,109,109,65 ),
  MuSigGTs( 121,113,113,113,65 ),
  MuSigGTs( 121,115,115,115,65 ),
  MuSigGTs( 123,117,117,117,65 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 3,1,65,65,3 ),
  MuSigGTs( 7,1,65,65,7 ),
  MuSigGTs( 11,1,65,65,13 ),
  MuSigGTs( 17,1,65,65,17 ),
  MuSigGTs( 23,1,1,65,23 ),
  MuSigGTs( 29,5,5,65,29 ),
  MuSigGTs( 39,9,9,9,35 ),
  MuSigGTs( 47,13,13,13,39 ),
  MuSigGTs( 57,21,21,21,45 ),
  MuSigGTs( 65,29,29,29,49 ),
  MuSigGTs( 71,39,39,39,51 ),
  MuSigGTs( 79,49,49,49,55 ),
  MuSigGTs( 85,59,59,59,57 ),
  MuSigGTs( 91,65,65,65,57 ),
  MuSigGTs( 99,73,73,73,59 ),
  MuSigGTs( 103,81,81,81,61 ),
  MuSigGTs( 107,87,87,87,61 ),
  MuSigGTs( 111,95,95,95,63 ),
  MuSigGTs( 113,99,99,99,63 ),
  MuSigGTs( 117,105,105,105,63 ),
  MuSigGTs( 119,109,109,109,63 ),
  MuSigGTs( 121,113,113,113,63 ),
  MuSigGTs( 121,115,115,115,65 ),
  MuSigGTs( 123,117,117,117,65 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 3,1,65,65,1 ),
  MuSigGTs( 7,1,65,65,1 ),
  MuSigGTs( 11,1,65,65,3 ),
  MuSigGTs( 15,1,65,65,5 ),
  MuSigGTs( 21,1,1,65,9 ),
  MuSigGTs( 27,1,1,65,15 ),
  MuSigGTs( 33,5,5,65,21 ),
  MuSigGTs( 41,9,9,9,27 ),
  MuSigGTs( 51,13,13,13,33 ),
  MuSigGTs( 59,21,21,21,37 ),
  MuSigGTs( 65,29,29,29,43 ),
  MuSigGTs( 71,39,39,39,47 ),
  MuSigGTs( 79,49,49,49,51 ),
  MuSigGTs( 87,59,59,59,53 ),
  MuSigGTs( 93,65,65,65,55 ),
  MuSigGTs( 99,73,73,73,57 ),
  MuSigGTs( 103,81,81,81,59 ),
  MuSigGTs( 107,87,87,87,61 ),
  MuSigGTs( 111,95,95,95,61 ),
  MuSigGTs( 113,99,99,99,63 ),
  MuSigGTs( 117,105,105,105,63 ),
  MuSigGTs( 119,109,109,109,63 ),
  MuSigGTs( 121,113,113,113,63 ),
  MuSigGTs( 121,115,115,115,63 ),
  MuSigGTs( 123,117,117,117,65 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 3,1,65,65,1 ),
  MuSigGTs( 5,1,65,65,1 ),
  MuSigGTs( 7,1,65,65,1 ),
  MuSigGTs( 11,1,65,65,1 ),
  MuSigGTs( 17,1,65,65,1 ),
  MuSigGTs( 21,1,65,65,3 ),
  MuSigGTs( 27,1,1,65,5 ),
  MuSigGTs( 33,1,1,65,9 ),
  MuSigGTs( 39,5,5,65,15 ),
  MuSigGTs( 47,9,9,9,21 ),
  MuSigGTs( 55,13,13,13,27 ),
  MuSigGTs( 63,21,21,21,31 ),
  MuSigGTs( 67,29,29,29,37 ),
  MuSigGTs( 73,39,39,39,41 ),
  MuSigGTs( 79,49,49,49,47 ),
  MuSigGTs( 87,59,59,59,49 ),
  MuSigGTs( 93,65,65,65,53 ),
  MuSigGTs( 99,73,73,73,55 ),
  MuSigGTs( 103,81,81,81,57 ),
  MuSigGTs( 107,87,87,87,59 ),
  MuSigGTs( 111,95,95,95,59 ),
  MuSigGTs( 115,99,99,99,61 ),
  MuSigGTs( 117,105,105,105,63 ),
  MuSigGTs( 119,109,109,109,63 ),
  MuSigGTs( 121,113,113,113,63 ),
  MuSigGTs( 123,115,115,115,63 ),
  MuSigGTs( 123,117,117,117,63 ),
  MuSigGTs( 125,119,119,119,65 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 3,1,65,65,1 ),
  MuSigGTs( 5,1,65,65,1 ),
  MuSigGTs( 9,1,65,65,1 ),
  MuSigGTs( 11,1,65,65,1 ),
  MuSigGTs( 17,1,65,65,1 ),
  MuSigGTs( 21,1,65,65,1 ),
  MuSigGTs( 27,1,65,65,1 ),
  MuSigGTs( 31,1,65,65,1 ),
  MuSigGTs( 37,1,1,65,3 ),
  MuSigGTs( 41,1,1,65,5 ),
  MuSigGTs( 47,5,5,65,9 ),
  MuSigGTs( 53,9,9,9,15 ),
  MuSigGTs( 59,13,13,13,21 ),
  MuSigGTs( 65,21,21,21,27 ),
  MuSigGTs( 69,29,29,29,33 ),
  MuSigGTs( 75,39,39,39,37 ),
  MuSigGTs( 81,49,49,49,43 ),
  MuSigGTs( 89,59,59,59,47 ),
  MuSigGTs( 93,65,65,65,49 ),
  MuSigGTs( 99,73,73,73,53 ),
  MuSigGTs( 103,81,81,81,55 ),
  MuSigGTs( 107,87,87,87,57 ),
  MuSigGTs( 111,95,95,95,59 ),
  MuSigGTs( 115,99,99,99,61 ),
  MuSigGTs( 117,105,105,105,61 ),
  MuSigGTs( 119,109,109,109,63 ),
  MuSigGTs( 121,113,113,113,63 ),
  MuSigGTs( 123,115,115,115,63 ),
  MuSigGTs( 123,117,117,117,63 ),
  MuSigGTs( 125,119,119,119,63 ),
  MuSigGTs( 125,121,121,121,65 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,65,65,65,65 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 1,1,65,65,1 ),
  MuSigGTs( 3,1,65,65,1 ),
  MuSigGTs( 7,1,65,65,1 ),
  MuSigGTs( 9,1,65,65,1 ),
  MuSigGTs( 13,1,65,65,1 ),
  MuSigGTs( 19,1,65,65,1 ),
  MuSigGTs( 23,1,65,65,1 ),
  MuSigGTs( 29,1,65,65,1 ),
  MuSigGTs( 33,1,65,65,1 ),
  MuSigGTs( 37,1,65,65,1 ),
  MuSigGTs( 41,1,65,65,1 ),
  MuSigGTs( 45,1,65,65,1 ),
  MuSigGTs( 49,1,1,65,1 ),
  MuSigGTs( 53,1,1,65,3 ),
  MuSigGTs( 57,5,5,65,7 ),
  MuSigGTs( 61,9,9,9,11 ),
  MuSigGTs( 65,13,13,13,17 ),
  MuSigGTs( 67,21,21,21,23 ),
  MuSigGTs( 73,29,29,29,27 ),
  MuSigGTs( 79,39,39,39,33 ),
  MuSigGTs( 83,49,49,49,39 ),
  MuSigGTs( 89,59,59,59,43 ),
  MuSigGTs( 95,65,65,65,47 ),
  MuSigGTs( 99,73,73,73,51 ),
  MuSigGTs( 105,81,81,81,53 ),
  MuSigGTs( 107,87,87,87,55 ),
  MuSigGTs( 111,95,95,95,57 ),
  MuSigGTs( 115,99,99,99,59 ),
  MuSigGTs( 117,105,105,105,61 ),
  MuSigGTs( 119,109,109,109,61 ),
  MuSigGTs( 121,113,113,113,63 ),
  MuSigGTs( 123,115,115,115,63 ),
  MuSigGTs( 123,117,117,117,63 ),
  MuSigGTs( 125,119,119,119,63 ),
  MuSigGTs( 125,121,121,121,63 ),
  MuSigGTs( 125,123,123,123,65 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
{
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,65,65,1 ),
  MuSigGTs( 65,1,1,65,1 ),
  MuSigGTs( 65,1,1,65,1 ),
  MuSigGTs( 65,1,1,65,1 ),
  MuSigGTs( 65,5,5,65,5 ),
  MuSigGTs( 67,9,9,9,7 ),
  MuSigGTs( 69,13,13,13,13 ),
  MuSigGTs( 73,21,21,21,19 ),
  MuSigGTs( 75,29,29,29,25 ),
  MuSigGTs( 81,39,39,39,29 ),
  MuSigGTs( 87,49,49,49,35 ),
  MuSigGTs( 91,59,59,59,41 ),
  MuSigGTs( 95,65,65,65,45 ),
  MuSigGTs( 101,73,73,73,49 ),
  MuSigGTs( 105,81,81,81,51 ),
  MuSigGTs( 109,87,87,87,55 ),
  MuSigGTs( 113,95,95,95,57 ),
  MuSigGTs( 115,99,99,99,57 ),
  MuSigGTs( 117,105,105,105,59 ),
  MuSigGTs( 119,109,109,109,61 ),
  MuSigGTs( 121,113,113,113,61 ),
  MuSigGTs( 123,115,115,115,63 ),
  MuSigGTs( 123,117,117,117,63 ),
  MuSigGTs( 125,119,119,119,63 ),
  MuSigGTs( 125,121,121,121,63 ),
  MuSigGTs( 125,123,123,123,63 ),
  MuSigGTs( 127,123,123,123,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,125,125,125,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
  MuSigGTs( 127,127,127,127,65 ),
},
};
