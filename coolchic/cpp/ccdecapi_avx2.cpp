/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/


#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <pybind11/stl.h>
#include <string>

// We have a complete decoder as the API.

// encode latents layer to a file.
int cc_decode_avx2(
    std::string &in_bitstream_filename,
    std::string &out_ppm_filename,
    int output_bitdepth = 0,
    int output_chroma_format = 0,
    int verbosity = 0);

PYBIND11_MODULE(ccdecapi_avx2, m) {
    m.doc() = "ccdecoding"; // optional module docstring
    m.def("cc_decode_avx2", &cc_decode_avx2, "decode a bitstream");
}

#ifndef CCDECAPI_AVX2
#define CCDECAPI_AVX2
#endif
#include "ccdecapi.cpp"
#undef CCDECAPI_AVX2
