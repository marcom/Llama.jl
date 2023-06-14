module LibLlama

using llama_cpp_jll
export llama_cpp_jll

using CEnum

mutable struct llama_context end

const llama_token = Cint

struct llama_token_data
    id::llama_token
    logit::Cfloat
    p::Cfloat
end

struct llama_token_data_array
    data::Ptr{llama_token_data}
    size::Csize_t
    sorted::Bool
end

# typedef void ( * llama_progress_callback ) ( float progress , void * ctx )
const llama_progress_callback = Ptr{Cvoid}

struct llama_context_params
    n_ctx::Cint
    n_gpu_layers::Cint
    seed::Cint
    f16_kv::Bool
    logits_all::Bool
    vocab_only::Bool
    use_mmap::Bool
    use_mlock::Bool
    embedding::Bool
    progress_callback::llama_progress_callback
    progress_callback_user_data::Ptr{Cvoid}
end

@cenum llama_ftype::UInt32 begin
    LLAMA_FTYPE_ALL_F32 = 0
    LLAMA_FTYPE_MOSTLY_F16 = 1
    LLAMA_FTYPE_MOSTLY_Q4_0 = 2
    LLAMA_FTYPE_MOSTLY_Q4_1 = 3
    LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
    LLAMA_FTYPE_MOSTLY_Q8_0 = 7
    LLAMA_FTYPE_MOSTLY_Q5_0 = 8
    LLAMA_FTYPE_MOSTLY_Q5_1 = 9
end

# no prototype is found for this function at llama.h:88:43, please use with caution
"""
    llama_context_default_params()


### Prototype
```c
struct llama_context_params llama_context_default_params();
```
"""
function llama_context_default_params()
    ccall((:llama_context_default_params, libllama), llama_context_params, ())
end

# no prototype is found for this function at llama.h:90:20, please use with caution
"""
    llama_mmap_supported()


### Prototype
```c
bool llama_mmap_supported();
```
"""
function llama_mmap_supported()
    ccall((:llama_mmap_supported, libllama), Bool, ())
end

# no prototype is found for this function at llama.h:91:20, please use with caution
"""
    llama_mlock_supported()


### Prototype
```c
bool llama_mlock_supported();
```
"""
function llama_mlock_supported()
    ccall((:llama_mlock_supported, libllama), Bool, ())
end

"""
    llama_init_from_file(path_model, params)


### Prototype
```c
struct llama_context * llama_init_from_file( const char * path_model, struct llama_context_params params);
```
"""
function llama_init_from_file(path_model, params)
    ccall((:llama_init_from_file, libllama), Ptr{llama_context}, (Ptr{Cchar}, llama_context_params), path_model, params)
end

"""
    llama_free(ctx)


### Prototype
```c
void llama_free(struct llama_context * ctx);
```
"""
function llama_free(ctx)
    ccall((:llama_free, libllama), Cvoid, (Ptr{llama_context},), ctx)
end

"""
    llama_model_quantize(fname_inp, fname_out, ftype, nthread)


### Prototype
```c
int llama_model_quantize( const char * fname_inp, const char * fname_out, enum llama_ftype ftype, int nthread);
```
"""
function llama_model_quantize(fname_inp, fname_out, ftype, nthread)
    ccall((:llama_model_quantize, libllama), Cint, (Ptr{Cchar}, Ptr{Cchar}, llama_ftype, Cint), fname_inp, fname_out, ftype, nthread)
end

"""
    llama_apply_lora_from_file(ctx, path_lora, path_base_model, n_threads)


### Prototype
```c
int llama_apply_lora_from_file( struct llama_context * ctx, const char * path_lora, const char * path_base_model, int n_threads);
```
"""
function llama_apply_lora_from_file(ctx, path_lora, path_base_model, n_threads)
    ccall((:llama_apply_lora_from_file, libllama), Cint, (Ptr{llama_context}, Ptr{Cchar}, Ptr{Cchar}, Cint), ctx, path_lora, path_base_model, n_threads)
end

"""
    llama_get_kv_cache_token_count(ctx)


### Prototype
```c
int llama_get_kv_cache_token_count(const struct llama_context * ctx);
```
"""
function llama_get_kv_cache_token_count(ctx)
    ccall((:llama_get_kv_cache_token_count, libllama), Cint, (Ptr{llama_context},), ctx)
end

"""
    llama_set_rng_seed(ctx, seed)


### Prototype
```c
void llama_set_rng_seed(struct llama_context * ctx, int seed);
```
"""
function llama_set_rng_seed(ctx, seed)
    ccall((:llama_set_rng_seed, libllama), Cvoid, (Ptr{llama_context}, Cint), ctx, seed)
end

"""
    llama_get_state_size(ctx)


### Prototype
```c
size_t llama_get_state_size(const struct llama_context * ctx);
```
"""
function llama_get_state_size(ctx)
    ccall((:llama_get_state_size, libllama), Csize_t, (Ptr{llama_context},), ctx)
end

"""
    llama_copy_state_data(ctx, dst)


### Prototype
```c
size_t llama_copy_state_data(struct llama_context * ctx, uint8_t * dst);
```
"""
function llama_copy_state_data(ctx, dst)
    ccall((:llama_copy_state_data, libllama), Csize_t, (Ptr{llama_context}, Ptr{UInt8}), ctx, dst)
end

"""
    llama_set_state_data(ctx, src)


### Prototype
```c
size_t llama_set_state_data(struct llama_context * ctx, const uint8_t * src);
```
"""
function llama_set_state_data(ctx, src)
    ccall((:llama_set_state_data, libllama), Csize_t, (Ptr{llama_context}, Ptr{UInt8}), ctx, src)
end

"""
    llama_load_session_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)


### Prototype
```c
bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
```
"""
function llama_load_session_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)
    ccall((:llama_load_session_file, libllama), Bool, (Ptr{llama_context}, Ptr{Cchar}, Ptr{llama_token}, Csize_t, Ptr{Csize_t}), ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)
end

"""
    llama_save_session_file(ctx, path_session, tokens, n_token_count)


### Prototype
```c
bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count);
```
"""
function llama_save_session_file(ctx, path_session, tokens, n_token_count)
    ccall((:llama_save_session_file, libllama), Bool, (Ptr{llama_context}, Ptr{Cchar}, Ptr{llama_token}, Csize_t), ctx, path_session, tokens, n_token_count)
end

"""
    llama_eval(ctx, tokens, n_tokens, n_past, n_threads)


### Prototype
```c
int llama_eval( struct llama_context * ctx, const llama_token * tokens, int n_tokens, int n_past, int n_threads);
```
"""
function llama_eval(ctx, tokens, n_tokens, n_past, n_threads)
    ccall((:llama_eval, libllama), Cint, (Ptr{llama_context}, Ptr{llama_token}, Cint, Cint, Cint), ctx, tokens, n_tokens, n_past, n_threads)
end

"""
    llama_tokenize(ctx, text, tokens, n_max_tokens, add_bos)


### Prototype
```c
int llama_tokenize( struct llama_context * ctx, const char * text, llama_token * tokens, int n_max_tokens, bool add_bos);
```
"""
function llama_tokenize(ctx, text, tokens, n_max_tokens, add_bos)
    ccall((:llama_tokenize, libllama), Cint, (Ptr{llama_context}, Ptr{Cchar}, Ptr{llama_token}, Cint, Bool), ctx, text, tokens, n_max_tokens, add_bos)
end

"""
    llama_n_vocab(ctx)


### Prototype
```c
int llama_n_vocab(const struct llama_context * ctx);
```
"""
function llama_n_vocab(ctx)
    ccall((:llama_n_vocab, libllama), Cint, (Ptr{llama_context},), ctx)
end

"""
    llama_n_ctx(ctx)


### Prototype
```c
int llama_n_ctx (const struct llama_context * ctx);
```
"""
function llama_n_ctx(ctx)
    ccall((:llama_n_ctx, libllama), Cint, (Ptr{llama_context},), ctx)
end

"""
    llama_n_embd(ctx)


### Prototype
```c
int llama_n_embd (const struct llama_context * ctx);
```
"""
function llama_n_embd(ctx)
    ccall((:llama_n_embd, libllama), Cint, (Ptr{llama_context},), ctx)
end

"""
    llama_get_logits(ctx)


### Prototype
```c
float * llama_get_logits(struct llama_context * ctx);
```
"""
function llama_get_logits(ctx)
    ccall((:llama_get_logits, libllama), Ptr{Cfloat}, (Ptr{llama_context},), ctx)
end

"""
    llama_get_embeddings(ctx)


### Prototype
```c
float * llama_get_embeddings(struct llama_context * ctx);
```
"""
function llama_get_embeddings(ctx)
    ccall((:llama_get_embeddings, libllama), Ptr{Cfloat}, (Ptr{llama_context},), ctx)
end

"""
    llama_token_to_str(ctx, token)


### Prototype
```c
const char * llama_token_to_str(const struct llama_context * ctx, llama_token token);
```
"""
function llama_token_to_str(ctx, token)
    ccall((:llama_token_to_str, libllama), Ptr{Cchar}, (Ptr{llama_context}, llama_token), ctx, token)
end

# no prototype is found for this function at llama.h:189:27, please use with caution
"""
    llama_token_bos()


### Prototype
```c
llama_token llama_token_bos();
```
"""
function llama_token_bos()
    ccall((:llama_token_bos, libllama), llama_token, ())
end

# no prototype is found for this function at llama.h:190:27, please use with caution
"""
    llama_token_eos()


### Prototype
```c
llama_token llama_token_eos();
```
"""
function llama_token_eos()
    ccall((:llama_token_eos, libllama), llama_token, ())
end

# no prototype is found for this function at llama.h:191:27, please use with caution
"""
    llama_token_nl()


### Prototype
```c
llama_token llama_token_nl();
```
"""
function llama_token_nl()
    ccall((:llama_token_nl, libllama), llama_token, ())
end

"""
    llama_sample_repetition_penalty(ctx, candidates, last_tokens, last_tokens_size, penalty)

@details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
### Prototype
```c
void llama_sample_repetition_penalty(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float penalty);
```
"""
function llama_sample_repetition_penalty(ctx, candidates, last_tokens, last_tokens_size, penalty)
    ccall((:llama_sample_repetition_penalty, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Ptr{llama_token}, Csize_t, Cfloat), ctx, candidates, last_tokens, last_tokens_size, penalty)
end

"""
    llama_sample_frequency_and_presence_penalties(ctx, candidates, last_tokens, last_tokens_size, alpha_frequency, alpha_presence)

@details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
### Prototype
```c
void llama_sample_frequency_and_presence_penalties(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float alpha_frequency, float alpha_presence);
```
"""
function llama_sample_frequency_and_presence_penalties(ctx, candidates, last_tokens, last_tokens_size, alpha_frequency, alpha_presence)
    ccall((:llama_sample_frequency_and_presence_penalties, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Ptr{llama_token}, Csize_t, Cfloat, Cfloat), ctx, candidates, last_tokens, last_tokens_size, alpha_frequency, alpha_presence)
end

"""
    llama_sample_softmax(ctx, candidates)

@details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
### Prototype
```c
void llama_sample_softmax(struct llama_context * ctx, llama_token_data_array * candidates);
```
"""
function llama_sample_softmax(ctx, candidates)
    ccall((:llama_sample_softmax, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}), ctx, candidates)
end

"""
    llama_sample_top_k(ctx, candidates, k, min_keep)

@details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
### Prototype
```c
void llama_sample_top_k(struct llama_context * ctx, llama_token_data_array * candidates, int k, size_t min_keep);
```
"""
function llama_sample_top_k(ctx, candidates, k, min_keep)
    ccall((:llama_sample_top_k, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cint, Csize_t), ctx, candidates, k, min_keep)
end

"""
    llama_sample_top_p(ctx, candidates, p, min_keep)

@details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
### Prototype
```c
void llama_sample_top_p(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);
```
"""
function llama_sample_top_p(ctx, candidates, p, min_keep)
    ccall((:llama_sample_top_p, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat, Csize_t), ctx, candidates, p, min_keep)
end

"""
    llama_sample_tail_free(ctx, candidates, z, min_keep)

@details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
### Prototype
```c
void llama_sample_tail_free(struct llama_context * ctx, llama_token_data_array * candidates, float z, size_t min_keep);
```
"""
function llama_sample_tail_free(ctx, candidates, z, min_keep)
    ccall((:llama_sample_tail_free, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat, Csize_t), ctx, candidates, z, min_keep)
end

"""
    llama_sample_typical(ctx, candidates, p, min_keep)

@details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
### Prototype
```c
void llama_sample_typical(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);
```
"""
function llama_sample_typical(ctx, candidates, p, min_keep)
    ccall((:llama_sample_typical, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat, Csize_t), ctx, candidates, p, min_keep)
end

"""
    llama_sample_temperature(ctx, candidates, temp)


### Prototype
```c
void llama_sample_temperature(struct llama_context * ctx, llama_token_data_array * candidates, float temp);
```
"""
function llama_sample_temperature(ctx, candidates, temp)
    ccall((:llama_sample_temperature, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat), ctx, candidates, temp)
end

"""
    llama_sample_token_mirostat(ctx, candidates, tau, eta, m, mu)

@details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
@param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
@param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
@param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
@param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
@param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
### Prototype
```c
llama_token llama_sample_token_mirostat(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, int m, float * mu);
```
"""
function llama_sample_token_mirostat(ctx, candidates, tau, eta, m, mu)
    ccall((:llama_sample_token_mirostat, libllama), llama_token, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat, Cfloat, Cint, Ptr{Cfloat}), ctx, candidates, tau, eta, m, mu)
end

"""
    llama_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu)

@details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
@param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
@param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
@param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
@param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
### Prototype
```c
llama_token llama_sample_token_mirostat_v2(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, float * mu);
```
"""
function llama_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu)
    ccall((:llama_sample_token_mirostat_v2, libllama), llama_token, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat, Cfloat, Ptr{Cfloat}), ctx, candidates, tau, eta, mu)
end

"""
    llama_sample_token_greedy(ctx, candidates)

@details Selects the token with the highest probability.
### Prototype
```c
llama_token llama_sample_token_greedy(struct llama_context * ctx, llama_token_data_array * candidates);
```
"""
function llama_sample_token_greedy(ctx, candidates)
    ccall((:llama_sample_token_greedy, libllama), llama_token, (Ptr{llama_context}, Ptr{llama_token_data_array}), ctx, candidates)
end

"""
    llama_sample_token(ctx, candidates)

@details Randomly selects a token from the candidates based on their probabilities.
### Prototype
```c
llama_token llama_sample_token(struct llama_context * ctx, llama_token_data_array * candidates);
```
"""
function llama_sample_token(ctx, candidates)
    ccall((:llama_sample_token, libllama), llama_token, (Ptr{llama_context}, Ptr{llama_token_data_array}), ctx, candidates)
end

"""
    llama_print_timings(ctx)


### Prototype
```c
void llama_print_timings(struct llama_context * ctx);
```
"""
function llama_print_timings(ctx)
    ccall((:llama_print_timings, libllama), Cvoid, (Ptr{llama_context},), ctx)
end

"""
    llama_reset_timings(ctx)


### Prototype
```c
void llama_reset_timings(struct llama_context * ctx);
```
"""
function llama_reset_timings(ctx)
    ccall((:llama_reset_timings, libllama), Cvoid, (Ptr{llama_context},), ctx)
end

"""
    llama_print_system_info()


### Prototype
```c
const char * llama_print_system_info(void);
```
"""
function llama_print_system_info()
    ccall((:llama_print_system_info, libllama), Ptr{Cchar}, ())
end

const LLAMA_FILE_VERSION = 3

# Skipping MacroDefinition: LLAMA_FILE_MAGIC 'ggjt'

# Skipping MacroDefinition: LLAMA_FILE_MAGIC_UNVERSIONED 'ggml'

# Skipping MacroDefinition: LLAMA_SESSION_MAGIC 'ggsn'

const LLAMA_SESSION_VERSION = 1

# exports
const PREFIXES = ["llama_", "LLAMA_", "ggml_", "GGML_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
