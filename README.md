# Llama_cpp.jl

Julia interface to
[llama.cpp](https://github.com/ggerganov/llama.cpp), a C/C++ port of
Meta's [LLaMA](https://arxiv.org/abs/2302.13971) (a large language
model).

## Installation

Press `]` at the Julia REPL to enter pkg mode, then:

```
add https://github.com/marcom/Llama_cpp_jll.jl
add https://github.com/marcom/Llama_cpp.jl
```

This currently only works on Linux for i686, x86_64, and aarch64 (due
cross-compilation for other targets not yet working in the
`Llama_cpp_jll/build_tarballs.jl` script).
