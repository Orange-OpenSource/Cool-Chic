/*
    Software Name: Cool-Chic
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#ifndef COMMON_RANDOMNESS
#define COMMON_RANDOMNESS

class common_randomness{

    public:
        float grand();
        short grand16();
    private:
        unsigned long int seed = 18101995; // nice seed
};

#endif // COMMON_RANDOMNESS
