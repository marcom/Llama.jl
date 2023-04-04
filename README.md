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
on Linux, Mac, and FreeBSD on i686, x86_64, and aarch64 (note: only
tested on x86_64-linux so far).

## Downloading the model weights

You will need a file with quantized model weights, see
[llama.cpp](https://github.com/ggerganov/llama.cpp) for instructions.

The binary format of the model weights changed on March 30, 2023 with
the new mmap-friendly file format. There are instructions for the
python conversion scripts in the
[llama.cpp](https://github.com/ggerganov/llama.cpp) repo.


## REPL mode

The REPL mode is currently non-functional, but stay tuned!

## LibLlama

```julia
ctx = LlamaContext("./ggml-alpaca-7b-q4.bin")
```

### `generate`

This currently seems to produce gibberish.

```julia
generate(ctx, "Write me a hello world in python")  # => currently prints text to screen
```

### `logits`

```julia
logits(ctx)  # => Vector{Float32}, length ctx.n_vocab
```

### `tokenize`

```julia
tokenize(ctx, "Hello world")  # => Vector{Int32} (token_ids), variable length
```

## Running example executables from llama.cpp

```julia
using Llama

s = run_llama(model="./ggml-alpaca-7b-q4.bin", prompt="Hello", args=`-n 16`)

# use more threads
run_llama(model="./ggml-alpaca-7b-q4.bin", prompt="Hello", nthreads=4)

# print the help text with more options
run_llama(model="", prompt="", args=`-h`)
```

### Interactive chat mode

```julia
run_chat(model="./ggml-alpaca-7b-q4.bin", prompt="Hello chat mode", nthreads=4)
```
