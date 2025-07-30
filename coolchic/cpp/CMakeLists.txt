
add_executable(ccdec)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

if(WIN32)

    message(STATUS "[ERROR] Cool-chic decoder not yet implemented for Windows...")

# Check Apple first, then UNIX (Apple + Linux) so that if we enter the UNIX if
# it means that we're on Linux.
elseif(APPLE)

    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")

        # Changes when compiling for arm64 Apple Mac:
        #   - Remove all *_avx2.cpp and *_avx512.cpp files
        #   - Remove the -mfa from the compilation options
        #   - Remove all the target_link_options... what is this for??
        #
        # It only compiles using g++/gcc, not clang which defaults to
        # an older version apparently?
        # cmake -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-13 -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-13 ..

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Wall -Winline")

        target_sources(ccdec PRIVATE ccdecapi.cpp cc-bitstream.cpp cc-contexts.cpp arm_cpu.cpp syn_cpu.cpp BitStream.cpp TDecBinCoderCABAC.cpp Contexts.cpp)

    else()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Winline")

        # For now, we compile  *_avx2.cpp and files, but they are
        # excluded from ccdec.cpp using quick & dirty #ifdef __APPLE__
        target_sources(ccdec PRIVATE ccdecapi.cpp cc-bitstream.cpp cc-contexts.cpp arm_cpu.cpp arm_avx2.cpp ups_cpu.cpp ups_avx2.cpp syn_cpu.cpp syn_avx2.cpp BitStream.cpp TDecBinCoderCABAC.cpp Contexts.cpp)

        set_source_files_properties(arm_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
        set_source_files_properties(ups_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
        set_source_files_properties(syn_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2")

    endif()

elseif(UNIX)

    message(STATUS "Architecture: Linux")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Wall -Winline -DCCDEC_EXE -DCCDECAPI_AVX2_OPTIONAL")

    target_sources(ccdec PRIVATE ccdecapi.cpp cc-bitstream.cpp cc-contexts.cpp frame-memory.cpp cc-frame-decoder.cpp arm_cpu.cpp arm_avx2.cpp ups_cpu.cpp ups_avx2.cpp syn_cpu.cpp syn_avx2.cpp warp.cpp BitStream.cpp TDecBinCoderCABAC.cpp Contexts.cpp)
    set(CMAKE_EXE_LINKER_FLAGS "-static")

    set_source_files_properties(arm_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
    set_source_files_properties(ups_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
    set_source_files_properties(syn_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2")

endif()

