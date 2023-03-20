using BinaryBuilder, Pkg

name = "Llama_cpp"
version = v"0.0.1"  # fake version number

# url = "https://github.com/ggerganov/llama.cpp"
# description = "Port of Facebook's LLaMA model in C/C++"

# TODO
# - is avx2 capability detected at run-time (via cpuid) or hardcoded
#   at build-time?
# - only {i686,x86_64,aarch64}-linux work at the moment
#   build failures:
#   - windows: not supported yet according to Makefile (2023.03.20)
#   - darwin: undeclared identifier ftello
#   - powerpc64le: Makefile assumes ppc is big-endian (2023.03.20)
#   - armv6l: disabled, don't know how to discern armv6l / armv7l in build script
#   - arm7vl: compile fail

sources = [
    # 2023.03.20, https://github.com/ggerganov/llama.cpp/releases/tag/master-074bea2
    # fake version number (used for this _jll) = 0.0.1
    GitSource("https://github.com/ggerganov/llama.cpp/",
              "074bea2eb1f1349a0118239c4152914aecaa1be4";
              unpack_target="llama.cpp"),
    DirectorySource("./bundled"),
]

script = raw"""
cd $WORKSPACE/srcdir/llama.cpp*

atomic_patch -p1 ../patches/fix-makefile-and-missing-clock_gettime.patch

UNAME_S=
case "${target}" in
    *-linux-*)
        UNAME_S=Linux
        ;;
    *-apple-darwin*)
        UNAME_S=Darwin
        ;;
    *-w64-mingw32*)
        UNAME_S=Windows
        ;;
    *-freebsd*)
        UNAME_S=FreeBSD
        ;;
esac

TARGET=$target
echo "TARGET=$TARGET"

UNAME_P=
UNAME_M=
case "${target}" in
    i686-*)
        UNAME_P=i686
        UNAME_M=i686
        ;;
    x86_64-*)
        UNAME_P=x86_64
        UNAME_M=x86_64
        ;;
    aarch64-*)
        UNAME_P=arm
        UNAME_M=arm64
        ;;
    arm-*)
        UNAME_P=arm
        # we assume armv7l
        UNAME_M=armv7l
        ;;
esac

# needed only for old glibc?
LDFLAGS="-lrt"

make -j${nproc} \
    CC="${CC}" \
    CXX="${CXX}" \
    LDFLAGS="${LDFLAGS}" \
    UNAME_S="${UNAME_S}" \
    UNAME_P="${UNAME_P}" \
    UNAME_M="${UNAME_M}"

for prg in main quantize; do
    install -Dvm 755 "./${prg}" "${bindir}/${prg}${exeext}"
done

install_license LICENSE
"""

platforms = supported_platforms(; exclude = p -> !Sys.islinux(p) || arch(p) ∉ ["i686", "x86_64", "aarch64"])
platforms = expand_cxxstring_abis(platforms)

products = [
    ExecutableProduct("main", :main),
    ExecutableProduct("quantize", :quantize),
]

dependencies = Dependency[
]

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat="1.6", preferred_gcc_version = v"11")
