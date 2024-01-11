# Llama.jl

[![Build Status](https://github.com/marcom/Llama.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/marcom/Llama.jl/actions/workflows/ci.yml?query=branch%3Amain)

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
on Linux, Mac, and FreeBSD on i686, x86_64, and aarch64 (note: only
tested on x86_64-linux and aarch64-macos so far).

## Downloading the model weights

You will need a file with quantized model weights in the right format (GGUF).

You can either download the weights from the [HuggingFace Hub](https://huggingface.co) (search for "GGUF" to download the right format) or convert them from the original PyTorch weights (see [llama.cpp](https://github.com/ggerganov/llama.cpp) for instructions.)

Good weights to start with are the OpenChat weights, which are Apache 2.0 licensed and can be downloaded [here](https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF). Click on the tab "Files" and download one of the `*.gguf` files. We recommend the Q4_K_M version (~4.4GB).
In the future, there might be new releases of the OpenChat weights, so look you might want to check for new versions.

TODO: Add a note on how to download "https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106.Q4_K_M.gguf"

## Running example executables from llama.cpp

### Llama Text Generation

```julia
using Llama

s = run_llama(model="models/openchat-3.5-0106.Q4_K_M.gguf", prompt="Hello")

# Provide additional arguments to llama.cpp (check the documentation for more details or the help text below)
s = run_llama(model="models/openchat-3.5-0106.Q4_K_M.gguf", prompt="Hello", n_gpu_layers=0, args=`-n 16`)

# print the help text with more options
run_llama(model="", prompt="", args=`-h`)
```

### Interactive chat mode

```julia
run_chat(model="models/openchat-3.5-0106.Q4_K_M.gguf", prompt="Hello chat mode")
```

## REPL mode

The REPL mode is currently non-functional, but stay tuned!

## LibLlama

The `libllama` bindings are currently non-functional, but stay tuned!