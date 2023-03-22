using BinaryBuilder, Pkg

name = "Llama_cpp"
version = v"0.0.2"  # fake version number

# url = "https://github.com/ggerganov/llama.cpp"
# description = "Port of Facebook's LLaMA model in C/C++"

# TODO
# - is e.g. avx2 capability detected at run-time (via cpuid) or hardcoded
#   at build-time?
# - i686, x86_64, aarch64 work at the moment
#   missing architectures: powerpc64le, armv6l, arm7vl

sources = [
    # 2023.03.21, https://github.com/ggerganov/llama.cpp/releases/tag/master-8cf9f34
    # fake version number (used for this _jll) = 0.0.2
    GitSource("https://github.com/ggerganov/llama.cpp.git",
              "8cf9f34eddc124d4ab28f4d2fe8e99d574510bde"),
    DirectorySource("./bundled"),

    # # 2023.03.20, https://github.com/ggerganov/llama.cpp/releases/tag/master-074bea2
    # # fake version number (used for this _jll) = 0.0.1
    # GitSource("https://github.com/ggerganov/llama.cpp.git",
    #           "074bea2eb1f1349a0118239c4152914aecaa1be4";
    #           unpack_target="llama.cpp"),
    # DirectorySource("./bundled"),
]

script = raw"""
cd $WORKSPACE/srcdir/llama.cpp*

atomic_patch -p1 ../patches/cmake-remove-mcpu-native.patch

EXTRA_CMAKE_ARGS=
if [[ "${target}" == *-linux-* ]]; then
    atomic_patch -p1 ../patches/fix-for-clock_gettime-not-found.patch
    EXTRA_CMAKE_ARGS='-DCMAKE_EXE_LINKER_FLAGS="-lrt"'
fi

mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$prefix \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
    -DCMAKE_BUILD_TYPE=RELEASE \
    $EXTRA_CMAKE_ARGS
make -j${nproc}

# `make install` doesn't work (2023.03.21)
for prg in llama quantize; do
    install -Dvm 755 "./${prg}${exeext}" "${bindir}/${prg}${exeext}"
done

install_license ../LICENSE
"""

platforms = supported_platforms(; exclude = p -> arch(p) ∉ ["i686", "x86_64", "aarch64"])
platforms = expand_cxxstring_abis(platforms)

products = [
    ExecutableProduct("llama", :llama),
    ExecutableProduct("quantize", :quantize),
]

dependencies = Dependency[
]

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat="1.6", preferred_gcc_version = v"11")
