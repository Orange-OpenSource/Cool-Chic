/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
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
    std::string &out_ppm_filename);

PYBIND11_MODULE(ccdecapi_avx2, m) {
    m.doc() = "ccdecoding"; // optional module docstring
    m.def("cc_decode_avx2", &cc_decode_avx2, "decode a bitstream");
}

#define CCDECAPI_AVX2
#include "ccdecapi.hpp"
#undef CCDECAPI_AVX2
