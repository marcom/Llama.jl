# executables

function run_llama(;
    model::AbstractString,
    prompt::AbstractString = "",
    nthreads::Int = 1,
    args = ``)
    cmd = `$(llama_cpp_jll.main()) --model $model --prompt $prompt --threads $nthreads $args`
    return read(cmd, String)
end

function run_chat(;
    model::AbstractString,
    prompt::AbstractString = "",
    nthreads::Int = 1,
    args = ``)
    cmd = `$(llama_cpp_jll.main()) --model $model --prompt $prompt --threads $nthreads $args -ins`
    run(cmd)
end

"""
    run_server(; model::AbstractString, host::AbstractString="127.0.0.1", port::Int=8080, nthreads::Int=Threads.nthreads(), n_gpu_layers::Int=99, args=``)

Starts a simple HTTP server with the `model` provided.

Open `http://{host}:{port}` in your browser to interact with the model or use an HTTP client to send requests to the server.

Interrupt the server with `Ctrl+C`.

# Arguments
- `model`: path to the model to be used
- `host`: host address to bind to. Defaults to "127.0.0.1"
- `port`: port to listen on. Defaults to 8080
- `nthreads`: number of threads to use. Defaults to the number of available threads
- `n_gpu_layers`: number of layers to offload on the GPU. Requires more VRAM on your GPU but can speed up inference. Defaults to 0 (=no layers)

See the [full documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md) for more details.

# Example

```julia
using Llama

# Download a model from HuggingFace, eg, Phi-2.
# See details [here](https://huggingface.co/TheBloke/dolphin-2_6-phi-2-GGUF)
using Downloads
model = joinpath("models", "dolphin-2_6-phi-2.Q6_K.gguf")
mkpath(dirname(model))
Downloads.download("https://huggingface.co/TheBloke/dolphin-2_6-phi-2-GGUF/resolve/main/dolphin-2_6-phi-2.Q6_K.gguf", model)
# go make a cup of tea while you wait... this is a 2.3GB download

# Start the server
run_server(; model)
"""
function run_server(;
    model::AbstractString,
    host::AbstractString = "127.0.0.1",
    port::Int = 8080,
    nthreads::Int = Threads.nthreads(),
    n_gpu_layers::Int = 0,
    args = ``)
    cmd = `$(llama_cpp_jll.server()) --model $model --host $host --port $port --threads $nthreads --n-gpu-layers $n_gpu_layers $args`
    run(cmd)
end
