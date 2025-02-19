# LlamaCpp.jl

[![Build Status](https://github.com/marcom/LlamaCpp.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/marcom/LlamaCpp.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Julia interface to
[llama.cpp](https://github.com/ggerganov/llama.cpp), a C/C++ port of
Meta's [LLaMA](https://arxiv.org/abs/2302.13971) (a large language
model).


## Installation

Press `]` at the Julia REPL to enter pkg mode, then:

```
add https://github.com/marcom/LlamaCpp.jl
```

The `llama_cpp_jll.jl` package used behind the scenes currently works
on Linux, Mac, and FreeBSD on `i686`, `x86_64`, and `aarch64` (note: only
tested on `x86_64-linux` and `aarch64-macos` so far).

## Downloading the model weights

You will need a file with quantized model weights in the right format (GGUF).

You can either download the weights from the [HuggingFace Hub](https://huggingface.co) (search for "GGUF" to download the right format) or convert them from the original PyTorch weights (see [llama.cpp](https://github.com/ggerganov/llama.cpp) for instructions.)

Good weights to start with are the Llama3-family fine-tuned weights ([here](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) with a Llama-specific licence) or Qwen 2.5 family, which are Apache 2.0 licensed and can be downloaded [here](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF). Click on the tab "Files" and download one of the `*.gguf` files. We recommend the Q5_K_M version (~5.5GB).

In the future, there might be new releases, so you might want to check for new versions.

Once you have a `url` link to a `.gguf` file, you can simply download it via:

```julia
using LlamaCpp
# Example for a 360M parameter model (c. 0.3GB)
url = "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q5_K_S.gguf"
model = download_model(url)
# Output: "models/SmolLM2-360M-Instruct-Q5_K_S.gguf"
```

You can use the model variable directly in the `run_*` functions, like `run_server`.

## Running example executables from llama.cpp

### Simple HTTP Server

Server mode is the easiest way to get started with LlamaCpp.jl. It provides both an in-browser chat interface and an OpenAI-compatible chat completion endpoint (for packages like [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl)).

```julia
using LlamaCpp

# Use the `model` downloaded above
LlamaCpp.run_server(; model)
```

Just open the URL `http://127.0.0.1:10897` in your browser to see the chat interface or use GET requests to the `/v1/chat/completions` endpoint.

### Llama Text Generation

```julia
using LlamaCpp
model = "models/SmolLM2-360M-Instruct-Q5_K_S.gguf"

s = run_llama(; model, prompt="Hello")

# Provide additional arguments to llama.cpp (check the documentation for more details or the help text below)
s = run_llama(; model, prompt="Hello", n_gpu_layers=0, args=`-n 16`)

# print the help text with more options
run_llama(model="", prompt="", args=`-h`)
```

> [!TIP]
> If you're getting gibberish output, it's likely that the model requires a "prompt template" (ie, structure to how you provide your instructions). Review the model page on HF Hub to see how to use your model or use the server.


### Interactive chat mode

```julia
run_chat(; model, prompt="Hello chat mode")
```

## REPL mode

The REPL mode is currently non-functional, but stay tuned!

## LibLlama

The `libllama` bindings are currently non-functional, but stay tuned!
