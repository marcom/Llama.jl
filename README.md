# Llama_cpp.jl

Julia interface to
[llama.cpp](https://github.com/ggerganov/llama.cpp), a C/C++ port of
Meta's [LLaMA](https://arxiv.org/abs/2302.13971) (a large language
model).

## Installation

Press `]` at the Julia REPL to enter pkg mode, then:

```
add https://github.com/marcom/Llama_cpp.jl
```

The `llama_cpp_jll.jl` package used behind the scenes currently works
on Linux, Windows, Mac, and FreeBSD on i686, x86_64, and aarch64 (note: only
tested on x86_64-linux so far).

## Downloading the model files

You will need a quantized model file, see
[llama.cpp](https://github.com/ggerganov/llama.cpp) for instructions.

## Running a model

```julia
s = llama(model="./ggml-alpaca-7b-q4.bin", prompt="Hello", args=`-n 16`)

# use more threads
llama(model="./ggml-alpaca-7b-q4.bin", prompt="Hello", nthreads=4)

# print the help text with more options
llama(model="", prompt="", args=`-h`)
```

## Interactive chat mode

```julia
chat(model="./ggml-alpaca-7b-q4.bin", prompt="Hello chat mode", nthreads=4)
```
