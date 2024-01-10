# executables

"""
    run_llama(; model::AbstractString, prompt::AbstractString="", nthreads::Int=1, n_gpu_layers::Int=99, ctx_size::Int=512, args=``)

Runs `prompt` through the `model` provided and returns the result. This is a single-turn version of `run_chat`.

# Arguments
- `model`: path to the model to be used
- `prompt`: prompt to be used. Most models expected these to be formatted in a specific way. Defaults to an empty string
- `nthreads`: number of threads to use. Defaults to the number of available threads
- `n_gpu_layers`: number of layers to offload on the GPU (a.k.a. `ngl` in llama.cpp). Requires more VRAM on your GPU but can speed up inference.
  Set to 0 to run inference on CPU-only. Defaults to 99 (=practically all layers)
- `ctx_size`: context size, ie, how big can the prompt/inference be. Defaults to 512 (but most models allow 4,000 and more)

Note: If you get odd responses AND you're using an instruction-tuned ("fine-tuned"), it might be that the format of your prompt is not correct. 
See HuggingFace's model documentation for the correct prompt format or use a library that will do this for you (eg, PromptingTools.jl)

See also: `run_chat`, `run_server`
"""
function run_llama(; model::AbstractString, prompt::AbstractString="", nthreads::Int=Threads.nthreads(), n_gpu_layers::Int=99, ctx_size::Int=512, args=``)
    cmd = `$(llama_cpp_jll.main()) --model $model --prompt $prompt --threads $nthreads --n-gpu-layers $n_gpu_layers --ctx-size $ctx_size $args`
    # Provides the path to locate ggml-metal.metal file (must be provided separately)
    cmd = addenv(cmd, "GGML_METAL_PATH_RESOURCES" => joinpath(llama_cpp_jll.artifact_dir, "bin"))
    return read(cmd, String)
end

"""
    run_chat(; model::AbstractString, prompt::AbstractString="", nthreads::Int=Threads.nthreads(), n_gpu_layers::Int=99, ctx_size::Int=512, args=``)

Opens an interactive console for the `model`. 
`prompt`, as the first message, is often used to provide instruction about the upcoming interactions (eg, style, tone, roles).

Wait for model to reply and then type your response. Press `Enter` to send the message to the model.

Interrupt the console with `Ctrl+C`

# Arguments
- `model`: path to the model to be used
- `prompt`: prompt to be used. Most models expected these to be formatted in a specific way. Defaults to an empty string
- `nthreads`: number of threads to use. Defaults to the number of available threads
- `n_gpu_layers`: number of layers to offload on the GPU (a.k.a. `ngl` in llama.cpp). Requires more VRAM on your GPU but can speed up inference.
  Set to 0 to run inference on CPU-only. Defaults to 99 (=practically all layers)
- `ctx_size`: context size, ie, how big can the prompt/inference be. Defaults to 512 (but most models allow 4,000 and more)

Note: If you get odd responses AND you're using an instruction-tuned ("fine-tuned"), it might be that the format of your prompt is not correct. 
See HuggingFace's model documentation for the correct prompt format or use a library that will do this for you (eg, PromptingTools.jl)

See also: `run_chat`, `run_server`
"""
function run_chat(; model::AbstractString, prompt::AbstractString="", nthreads::Int=Threads.nthreads(), n_gpu_layers::Int=99, ctx_size::Int=512, args=``)
    cmd = `$(llama_cpp_jll.main()) --model $model --prompt $prompt --threads $nthreads --n-gpu-layers $n_gpu_layers --ctx-size $ctx_size $args -ins`
    # Provides the path to locate ggml-metal.metal file (must be provided separately)
    cmd = addenv(cmd, "GGML_METAL_PATH_RESOURCES" => joinpath(llama_cpp_jll.artifact_dir, "bin"))
    run(cmd)
end

"""
    run_server(; model::AbstractString, host::AbstractString="127.0.0.1", port::Int=519, nthreads::Int=Threads.nthreads(), 
    n_gpu_layers::Int=99, ctx_size::Int=512, args=``)

Starts a simple HTTP server with the `model` provided.

Open `http://{host}:{port}` in your browser to interact with the model or use an HTTP client to send requests to the server.

Interrupt the server with `Ctrl+C`.

# Arguments
- `model`: path to the model to be used
- `host`: host address to bind to. Defaults to "127.0.0.1"
- `port`: port to listen on. Defaults to 519 (Why 519? Check the sum of ASCII codes of the letters in "llama")
- `nthreads`: number of threads to use. Defaults to the number of available threads
- `n_gpu_layers`: number of layers to offload on the GPU (a.k.a. `ngl` in llama.cpp). Requires more VRAM on your GPU but can speed up inference.
  Set to 0 to run inference on CPU-only. Defaults to 99 (=practically all layers)
- `ctx_size`: context size, ie, how big can the prompt/inference be. Defaults to 512 (but most models allow 4,000 and more)

See the [full documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md) for more details.

# Example

```julia
using Llama

# Download a model from HuggingFace, eg, Phi-2.
# See details [here](https://huggingface.co/TheBloke/dolphin-2_6-phi-2-GGUF)
using Downloads
model = joinpath("models", "dolphin-2_6-phi-2.Q6_K.gguf")
mkpath(dirname(model)) # ensure the folder exists
Downloads.download("https://huggingface.co/TheBloke/dolphin-2_6-phi-2-GGUF/resolve/main/dolphin-2_6-phi-2.Q6_K.gguf", model)
# go make a cup of tea while you wait... this is a 2.3GB download

# Start the server
run_server(; model)
"""
function run_server(; model::AbstractString, host::AbstractString="127.0.0.1", port::Int=519, nthreads::Int=Threads.nthreads(), n_gpu_layers::Int=99, ctx_size::Int=512, args=``)
    cmd = `$(llama_cpp_jll.server()) --model $model --host $host --port $port --threads $nthreads --n-gpu-layers $n_gpu_layers --ctx-size $ctx_size $args`
    # Provides the path to locate ggml-metal.metal file (must be provided separately)
    cmd = addenv(cmd, "GGML_METAL_PATH_RESOURCES" => joinpath(llama_cpp_jll.artifact_dir, "bin"))
    run(cmd)
end
