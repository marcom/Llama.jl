module Llama

export run_llama, run_chat

import llama_cpp_jll

include("../lib/LibLlama.jl")
# llama_* types
import .LibLlama: llama_token
# llama_* functions
import .LibLlama: llama_context_default_params, llama_init_from_file,
    llama_n_vocab, llama_n_ctx, llama_get_logits, llama_free,
    llama_print_timings, llama_reset_timings, llama_tokenize,
    llama_eval

# executables

function run_llama(; model::AbstractString, prompt::AbstractString="", nthreads::Int=1, args=``)
    cmd = `$(llama_cpp_jll.main()) --model $model --prompt $prompt --threads $nthreads $args`
    return read(cmd, String)
end

function run_chat(; model::AbstractString, prompt::AbstractString="", nthreads::Int=1, args=``)
    cmd = `$(llama_cpp_jll.main()) --model $model --prompt $prompt --threads $nthreads $args -ins`
    run(cmd)
end

end # module Llama
