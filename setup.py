# Available at setup time due to pyproject.toml

import subprocess

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from sys import platform

__version__ = "3.3.0"

CCLIB_PATH = 'coolchic/CCLIB'

subprocess.call(f"rm -rf {CCLIB_PATH}/*", shell=True)
subprocess.call("rm -rf coolchic/coolchic.egg-info/*", shell=True)


# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension(
        "ccencapi",
        [
            "coolchic/cpp/ccencapi.cpp",
            "coolchic/cpp/TEncBinCoderCABAC.cpp",
            "coolchic/cpp/TDecBinCoderCABAC.cpp",
            "coolchic/cpp/Contexts.cpp",
            "coolchic/cpp/BitStream.cpp",
            "coolchic/cpp/cc-contexts.cpp"
        ],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=["-g", "-O3"],
    ),
    Pybind11Extension(
        "ccdecapi_cpu",
        [
            "coolchic/cpp/ccdecapi_cpu.cpp",
            "coolchic/cpp/cc-bitstream.cpp",
            "coolchic/cpp/cc-contexts.cpp",
            "coolchic/cpp/cc-frame-decoder.cpp",
            "coolchic/cpp/frame-memory.cpp",
            "coolchic/cpp/arm_cpu.cpp",
            "coolchic/cpp/syn_cpu.cpp",
            "coolchic/cpp/BitStream.cpp",
            "coolchic/cpp/TDecBinCoderCABAC.cpp",
            "coolchic/cpp/Contexts.cpp"],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__), ("CCDECAPI_CPU", "1")],
        extra_compile_args=["-g", "-O3"],
    ),
]

if platform != "darwin":
    ext_modules.append(
        Pybind11Extension(
            "ccdecapi_avx2",
            [
                "coolchic/cpp/ccdecapi_avx2.cpp",
                "coolchic/cpp/cc-bitstream.cpp",
                "coolchic/cpp/cc-contexts.cpp",
                "coolchic/cpp/cc-frame-decoder.cpp",
                "coolchic/cpp/frame-memory.cpp",
                "coolchic/cpp/arm_cpu.cpp",
                "coolchic/cpp/arm_avx2.cpp",
                "coolchic/cpp/ups_avx2.cpp",
                "coolchic/cpp/syn_cpu.cpp",
                "coolchic/cpp/syn_avx2.cpp",
                "coolchic/cpp/BitStream.cpp",
                "coolchic/cpp/TDecBinCoderCABAC.cpp",
                "coolchic/cpp/Contexts.cpp"
            ],
            # Example: passing in the version to the compiled code
            define_macros=[("VERSION_INFO", __version__), ("CCDECAPI_AVX2", "1")],
            extra_compile_args=["-g", "-O3", "-mavx2"],
        )
    )

setup(
    name="coolchic",
    version=__version__,
    author="Orange",
    author_email="theo.ladune@orange.com",
    url="https://github.com/Orange-OpenSource/Cool-Chic",
    description="Cool-Chic: lightweight neural video codec.",
    long_description="",
    ext_modules=ext_modules,
    extras_require={},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "torchvision",
        "matplotlib",
        "einops",
        "fvcore",
        "cmake",
        "ConfigArgParse",
        "psutil",
        "pytest",
        "pytest-order",
    ]
)

subprocess.call(f"mkdir -p {CCLIB_PATH}", shell=True)

subprocess.call(f"mv *.so {CCLIB_PATH}", shell=True)
