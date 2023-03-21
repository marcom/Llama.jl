module Llama_cpp

export llama, chat

import Llama_cpp_jll

# URLs taken from llama.cpp
#
# ggml-alpaca-7b-q4.bin
#
# curl -o ggml-alpaca-7b-q4.bin -C - https://gateway.estuary.tech/gw/ipfs/QmUp1UGeQFDqJKvtjbSYPBiZZKRjLp8shVP9hT8ZB9Ynv1
# curl -o ggml-alpaca-7b-q4.bin -C - https://ipfs.io/ipfs/QmUp1UGeQFDqJKvtjbSYPBiZZKRjLp8shVP9hT8ZB9Ynv1
# curl -o ggml-alpaca-7b-q4.bin -C - https://cloudflare-ipfs.com/ipfs/QmUp1UGeQFDqJKvtjbSYPBiZZKRjLp8shVP9hT8ZB9Ynv1

const URLs = Dict(
    "ggml-alpaca-7b-q4.bin" => [
        "https://gateway.estuary.tech/gw/ipfs/QmQ1bf2BTnYxq73MFJWu1B7bQ2UD6qG7D7YDCxhTndVkPC",
        "https://ipfs.io/ipfs/QmQ1bf2BTnYxq73MFJWu1B7bQ2UD6qG7D7YDCxhTndVkPC",
        "https://cloudflare-ipfs.com/ipfs/QmQ1bf2BTnYxq73MFJWu1B7bQ2UD6qG7D7YDCxhTndVkPC",
    ],
    "ggml2-alpaca-7b-q4.bin" => [
        "https://gateway.estuary.tech/gw/ipfs/QmUp1UGeQFDqJKvtjbSYPBiZZKRjLp8shVP9hT8ZB9Ynv1",
        "https://ipfs.io/ipfs/QmUp1UGeQFDqJKvtjbSYPBiZZKRjLp8shVP9hT8ZB9Ynv1",
        "https://cloudflare-ipfs.com/ipfs/QmUp1UGeQFDqJKvtjbSYPBiZZKRjLp8shVP9hT8ZB9Ynv1",
    ],
)

function llama(; model::AbstractString, prompt::AbstractString="", nthreads::Int=1, args=``)
    cmd = `$(Llama_cpp_jll.main()) --model $model --prompt $prompt --threads $nthreads $args`
    return read(cmd, String)
end

function chat(; model::AbstractString, prompt::AbstractString="", nthreads::Int=1, args=``)
    cmd = `$(Llama_cpp_jll.main()) --model $model --prompt $prompt --threads $nthreads $args -ins`
    run(cmd)
end

end # module Llama_cpp
