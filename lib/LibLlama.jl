module LibLlama

using llama_cpp_jll
export llama_cpp_jll

using CEnum

mutable struct llama_context end

const llama_token = Cint

struct llama_token_data
    id::llama_token
    p::Cfloat
    plog::Cfloat
end

# typedef void ( * llama_progress_callback ) ( double progress , void * ctx )
const llama_progress_callback = Ptr{Cvoid}

struct llama_context_params
    n_ctx::Cint
    n_parts::Cint
    seed::Cint
    f16_kv::Bool
    logits_all::Bool
    vocab_only::Bool
    use_mlock::Bool
    embedding::Bool
    progress_callback::llama_progress_callback
    progress_callback_user_data::Ptr{Cvoid}
end

# no prototype is found for this function at llama.h:67:43, please use with caution
function llama_context_default_params()
    ccall((:llama_context_default_params, libllama), llama_context_params, ())
end

function llama_init_from_file(path_model, params)
    ccall((:llama_init_from_file, libllama), Ptr{llama_context}, (Ptr{Cchar}, llama_context_params), path_model, params)
end

function llama_free(ctx)
    ccall((:llama_free, libllama), Cvoid, (Ptr{llama_context},), ctx)
end

function llama_model_quantize(fname_inp, fname_out, itype, qk)
    ccall((:llama_model_quantize, libllama), Cint, (Ptr{Cchar}, Ptr{Cchar}, Cint, Cint), fname_inp, fname_out, itype, qk)
end

function llama_eval(ctx, tokens, n_tokens, n_past, n_threads)
    ccall((:llama_eval, libllama), Cint, (Ptr{llama_context}, Ptr{llama_token}, Cint, Cint, Cint), ctx, tokens, n_tokens, n_past, n_threads)
end

function llama_tokenize(ctx, text, tokens, n_max_tokens, add_bos)
    ccall((:llama_tokenize, libllama), Cint, (Ptr{llama_context}, Ptr{Cchar}, Ptr{llama_token}, Cint, Bool), ctx, text, tokens, n_max_tokens, add_bos)
end

function llama_n_vocab(ctx)
    ccall((:llama_n_vocab, libllama), Cint, (Ptr{llama_context},), ctx)
end

function llama_n_ctx(ctx)
    ccall((:llama_n_ctx, libllama), Cint, (Ptr{llama_context},), ctx)
end

function llama_n_embd(ctx)
    ccall((:llama_n_embd, libllama), Cint, (Ptr{llama_context},), ctx)
end

function llama_get_logits(ctx)
    ccall((:llama_get_logits, libllama), Ptr{Cfloat}, (Ptr{llama_context},), ctx)
end

function llama_get_embeddings(ctx)
    ccall((:llama_get_embeddings, libllama), Ptr{Cfloat}, (Ptr{llama_context},), ctx)
end

function llama_token_to_str(ctx, token)
    ccall((:llama_token_to_str, libllama), Ptr{Cchar}, (Ptr{llama_context}, llama_token), ctx, token)
end

# no prototype is found for this function at llama.h:129:27, please use with caution
function llama_token_bos()
    ccall((:llama_token_bos, libllama), llama_token, ())
end

# no prototype is found for this function at llama.h:130:27, please use with caution
function llama_token_eos()
    ccall((:llama_token_eos, libllama), llama_token, ())
end

function llama_sample_top_p_top_k(ctx, last_n_tokens_data, last_n_tokens_size, top_k, top_p, temp, repeat_penalty)
    ccall((:llama_sample_top_p_top_k, libllama), llama_token, (Ptr{llama_context}, Ptr{llama_token}, Cint, Cint, Cdouble, Cdouble, Cdouble), ctx, last_n_tokens_data, last_n_tokens_size, top_k, top_p, temp, repeat_penalty)
end

function llama_print_timings(ctx)
    ccall((:llama_print_timings, libllama), Cvoid, (Ptr{llama_context},), ctx)
end

function llama_reset_timings(ctx)
    ccall((:llama_reset_timings, libllama), Cvoid, (Ptr{llama_context},), ctx)
end

function llama_print_system_info()
    ccall((:llama_print_system_info, libllama), Ptr{Cchar}, ())
end

const LLAMA_FILE_VERSION = 1

const LLAMA_FILE_MAGIC = 0x67676d66

const LLAMA_FILE_MAGIC_UNVERSIONED = 0x67676d6c

# exports
const PREFIXES = ["llama_", "LLAMA_", "ggml_", "GGML_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
