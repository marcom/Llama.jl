# Llama.jl

Julia interface to
[llama.cpp](https://github.com/ggerganov/llama.cpp), a C/C++ port of
Meta's [LLaMA](https://arxiv.org/abs/2302.13971) (a large language
model).

## Installation

Press `]` at the Julia REPL to enter pkg mode, then:

```
add https://github.com/marcom/Llama.jl
```

The `llama_cpp_jll.jl` package used behind the scenes currently works
on Linux, Windows, Mac, and FreeBSD on i686, x86_64, and aarch64 (note: only
tested on x86_64-linux so far).

## Downloading the model weights

You will need a file with quantized model weights, see
[llama.cpp](https://github.com/ggerganov/llama.cpp) for instructions.

The binary format of the model weights changed around March
20, 2023. There is a [conversion
script](https://gist.github.com/eiz/828bddec6162a023114ce19146cb2b82)
if you have older weights which cannot be loaded anymore. Note that
you have to use the `--n_parts 1` command-line option if you want to
use the 1-part model weights this conversion script produces.

## Running a model

```julia
using Llama

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
