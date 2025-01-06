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
int cc_decode_cpu(
    std::string &in_bitstream_filename,
    std::string &out_ppm_filename,
    int output_bitdepth = 0,
    int output_chroma_format = 0,
    int verbosity = 0);

PYBIND11_MODULE(ccdecapi_cpu, m) {
    m.doc() = "ccdecoding"; // optional module docstring
    m.def("cc_decode_cpu", &cc_decode_cpu, "decode a bitstream");
}

#ifndef CCDECAPI_CPU
#define CCDECAPI_CPU
#endif
#include "ccdecapi.cpp"
#undef CCDECAPI_CPU
