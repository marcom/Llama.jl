# Llama.jl

[![Build Status](https://github.com/marcom/Llama.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/marcom/Llama.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

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

Good weights to start with are the Dolphin-family fine-tuned weights, which are Apache 2.0 licensed and can be downloaded [here](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF). Click on the tab "Files" and download one of the `*.gguf` files. We recommend the Q4_K_M version (~4.4GB).

In the future, there might be new releases, so you might want to check for new versions.

Once you have a `url` link to a `.gguf` file, you can simply download it via:

```julia
using Llama
# Example for a 7Bn parameter model (c. 4.4GB)
url = "https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF/resolve/main/dolphin-2.6-mistral-7b-dpo.Q4_K_M.gguf"
model = download_model(url)
# Output: "models/dolphin-2.6-mistral-7b-dpo.Q4_K_M.gguf"
```

You can use the model variable directly in the `run_*` functions, like `run_server`.

## Running example executables from llama.cpp

### Simple HTTP Server

Server mode is the easiest way to get started with Llama.jl. It provides both an in-browser chat interface and an OpenAI-compatible chat completion endpoint (for packages like [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl)).

```julia
using Llama

# Use the `model` downloaded above
Llama.run_server(; model)
```

### Llama Text Generation

```julia
using Llama

s = run_llama(model="models/dolphin-2.6-mistral-7b-dpo.Q4_K_M.gguf", prompt="Hello")

# Provide additional arguments to llama.cpp (check the documentation for more details or the help text below)
s = run_llama(model="models/dolphin-2.6-mistral-7b-dpo.Q4_K_M.gguf", prompt="Hello", n_gpu_layers=0, args=`-n 16`)

# print the help text with more options
run_llama(model="", prompt="", args=`-h`)
```

> [!TIP]
> If you're getting gibberish output, it's likely that the model requires a "prompt template" (ie, structure to how you provide your instructions). Review the model page on HF Hub to see how to use your model or use the server.


### Interactive chat mode

```julia
run_chat(model="models/dolphin-2.6-mistral-7b-dpo.Q4_K_M.gguf", prompt="Hello chat mode")
```

## REPL mode

The REPL mode is currently non-functional, but stay tuned!

## LibLlama

The `libllama` bindings are currently non-functional, but stay tuned!
