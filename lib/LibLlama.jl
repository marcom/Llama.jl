module LibLlama

using llama_cpp_jll
export llama_cpp_jll

using CEnum

mutable struct llama_model end

mutable struct llama_context end

const llama_pos = Int32

const llama_token = Int32

const llama_seq_id = Int32

@cenum llama_vocab_type::UInt32 begin
    LLAMA_VOCAB_TYPE_SPM = 0
    LLAMA_VOCAB_TYPE_BPE = 1
end

@cenum llama_token_type::UInt32 begin
    LLAMA_TOKEN_TYPE_UNDEFINED = 0
    LLAMA_TOKEN_TYPE_NORMAL = 1
    LLAMA_TOKEN_TYPE_UNKNOWN = 2
    LLAMA_TOKEN_TYPE_CONTROL = 3
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4
    LLAMA_TOKEN_TYPE_UNUSED = 5
    LLAMA_TOKEN_TYPE_BYTE = 6
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
    LLAMA_FTYPE_MOSTLY_Q2_K = 10
    LLAMA_FTYPE_MOSTLY_Q3_K_S = 11
    LLAMA_FTYPE_MOSTLY_Q3_K_M = 12
    LLAMA_FTYPE_MOSTLY_Q3_K_L = 13
    LLAMA_FTYPE_MOSTLY_Q4_K_S = 14
    LLAMA_FTYPE_MOSTLY_Q4_K_M = 15
    LLAMA_FTYPE_MOSTLY_Q5_K_S = 16
    LLAMA_FTYPE_MOSTLY_Q5_K_M = 17
    LLAMA_FTYPE_MOSTLY_Q6_K = 18
    LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19
    LLAMA_FTYPE_GUESSED = 1024
end

@cenum llama_rope_scaling_type::Int32 begin
    LLAMA_ROPE_SCALING_UNSPECIFIED = -1
    LLAMA_ROPE_SCALING_NONE = 0
    LLAMA_ROPE_SCALING_LINEAR = 1
    LLAMA_ROPE_SCALING_YARN = 2
    LLAMA_ROPE_SCALING_MAX_VALUE = 2
end

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

# typedef bool ( * llama_progress_callback ) ( float progress , void * ctx )
const llama_progress_callback = Ptr{Cvoid}

struct llama_batch
    n_tokens::Int32
    token::Ptr{llama_token}
    embd::Ptr{Cfloat}
    pos::Ptr{llama_pos}
    n_seq_id::Ptr{Int32}
    seq_id::Ptr{Ptr{llama_seq_id}}
    logits::Ptr{Int8}
    all_pos_0::llama_pos
    all_pos_1::llama_pos
    all_seq_id::llama_seq_id
end

@cenum llama_model_kv_override_type::UInt32 begin
    LLAMA_KV_OVERRIDE_INT = 0
    LLAMA_KV_OVERRIDE_FLOAT = 1
    LLAMA_KV_OVERRIDE_BOOL = 2
end

struct llama_model_kv_override
    data::NTuple{144, UInt8}
end

function Base.getproperty(x::Ptr{llama_model_kv_override}, f::Symbol)
    f === :key && return Ptr{NTuple{128, Cchar}}(x + 0)
    f === :tag && return Ptr{llama_model_kv_override_type}(x + 128)
    f === :int_value && return Ptr{Int64}(x + 136)
    f === :float_value && return Ptr{Cdouble}(x + 136)
    f === :bool_value && return Ptr{Bool}(x + 136)
    return getfield(x, f)
end

function Base.getproperty(x::llama_model_kv_override, f::Symbol)
    r = Ref{llama_model_kv_override}(x)
    ptr = Base.unsafe_convert(Ptr{llama_model_kv_override}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{llama_model_kv_override}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct llama_model_params
    n_gpu_layers::Int32
    main_gpu::Int32
    tensor_split::Ptr{Cfloat}
    progress_callback::llama_progress_callback
    progress_callback_user_data::Ptr{Cvoid}
    kv_overrides::Ptr{llama_model_kv_override}
    vocab_only::Bool
    use_mmap::Bool
    use_mlock::Bool
end

@cenum ggml_type::UInt32 begin
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_I8 = 17
    GGML_TYPE_I16 = 18
    GGML_TYPE_I32 = 19
    GGML_TYPE_COUNT = 20
end

struct llama_context_params
    seed::UInt32
    n_ctx::UInt32
    n_batch::UInt32
    n_threads::UInt32
    n_threads_batch::UInt32
    rope_scaling_type::Int8
    rope_freq_base::Cfloat
    rope_freq_scale::Cfloat
    yarn_ext_factor::Cfloat
    yarn_attn_factor::Cfloat
    yarn_beta_fast::Cfloat
    yarn_beta_slow::Cfloat
    yarn_orig_ctx::UInt32
    type_k::ggml_type
    type_v::ggml_type
    mul_mat_q::Bool
    logits_all::Bool
    embedding::Bool
    offload_kqv::Bool
end

struct llama_model_quantize_params
    nthread::Int32
    ftype::llama_ftype
    allow_requantize::Bool
    quantize_output_tensor::Bool
    only_copy::Bool
    pure::Bool
end

mutable struct llama_grammar end

@cenum llama_gretype::UInt32 begin
    LLAMA_GRETYPE_END = 0
    LLAMA_GRETYPE_ALT = 1
    LLAMA_GRETYPE_RULE_REF = 2
    LLAMA_GRETYPE_CHAR = 3
    LLAMA_GRETYPE_CHAR_NOT = 4
    LLAMA_GRETYPE_CHAR_RNG_UPPER = 5
    LLAMA_GRETYPE_CHAR_ALT = 6
end

struct llama_grammar_element
    type::llama_gretype
    value::UInt32
end

struct llama_timings
    t_start_ms::Cdouble
    t_end_ms::Cdouble
    t_load_ms::Cdouble
    t_sample_ms::Cdouble
    t_p_eval_ms::Cdouble
    t_eval_ms::Cdouble
    n_sample::Int32
    n_p_eval::Int32
    n_eval::Int32
end

"""
    llama_model_default_params()


### Prototype
```c
struct llama_model_params llama_model_default_params(void);
```
"""
function llama_model_default_params()
    ccall((:llama_model_default_params, libllama), llama_model_params, ())
end

"""
    llama_context_default_params()


### Prototype
```c
struct llama_context_params llama_context_default_params(void);
```
"""
function llama_context_default_params()
    ccall((:llama_context_default_params, libllama), llama_context_params, ())
end

"""
    llama_model_quantize_default_params()


### Prototype
```c
struct llama_model_quantize_params llama_model_quantize_default_params(void);
```
"""
function llama_model_quantize_default_params()
    ccall((:llama_model_quantize_default_params, libllama), llama_model_quantize_params, ())
end

"""
    llama_backend_init(numa)


### Prototype
```c
void llama_backend_init(bool numa);
```
"""
function llama_backend_init(numa)
    ccall((:llama_backend_init, libllama), Cvoid, (Bool,), numa)
end

"""
    llama_backend_free()


### Prototype
```c
void llama_backend_free(void);
```
"""
function llama_backend_free()
    ccall((:llama_backend_free, libllama), Cvoid, ())
end

"""
    llama_load_model_from_file(path_model, params)


### Prototype
```c
struct llama_model * llama_load_model_from_file( const char * path_model, struct llama_model_params params);
```
"""
function llama_load_model_from_file(path_model, params)
    ccall((:llama_load_model_from_file, libllama), Ptr{llama_model}, (Ptr{Cchar}, llama_model_params), path_model, params)
end

"""
    llama_free_model(model)


### Prototype
```c
void llama_free_model(struct llama_model * model);
```
"""
function llama_free_model(model)
    ccall((:llama_free_model, libllama), Cvoid, (Ptr{llama_model},), model)
end

"""
    llama_new_context_with_model(model, params)


### Prototype
```c
struct llama_context * llama_new_context_with_model( struct llama_model * model, struct llama_context_params params);
```
"""
function llama_new_context_with_model(model, params)
    ccall((:llama_new_context_with_model, libllama), Ptr{llama_context}, (Ptr{llama_model}, llama_context_params), model, params)
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
    llama_time_us()


### Prototype
```c
int64_t llama_time_us(void);
```
"""
function llama_time_us()
    ccall((:llama_time_us, libllama), Int64, ())
end

"""
    llama_max_devices()


### Prototype
```c
int32_t llama_max_devices(void);
```
"""
function llama_max_devices()
    ccall((:llama_max_devices, libllama), Int32, ())
end

"""
    llama_mmap_supported()


### Prototype
```c
bool llama_mmap_supported (void);
```
"""
function llama_mmap_supported()
    ccall((:llama_mmap_supported, libllama), Bool, ())
end

"""
    llama_mlock_supported()


### Prototype
```c
bool llama_mlock_supported(void);
```
"""
function llama_mlock_supported()
    ccall((:llama_mlock_supported, libllama), Bool, ())
end

"""
    llama_get_model(ctx)


### Prototype
```c
const struct llama_model * llama_get_model(const struct llama_context * ctx);
```
"""
function llama_get_model(ctx)
    ccall((:llama_get_model, libllama), Ptr{llama_model}, (Ptr{llama_context},), ctx)
end

"""
    llama_n_ctx(ctx)


### Prototype
```c
uint32_t llama_n_ctx (const struct llama_context * ctx);
```
"""
function llama_n_ctx(ctx)
    ccall((:llama_n_ctx, libllama), UInt32, (Ptr{llama_context},), ctx)
end

"""
    llama_n_batch(ctx)


### Prototype
```c
uint32_t llama_n_batch (const struct llama_context * ctx);
```
"""
function llama_n_batch(ctx)
    ccall((:llama_n_batch, libllama), UInt32, (Ptr{llama_context},), ctx)
end

"""
    llama_vocab_type(model)


### Prototype
```c
enum llama_vocab_type llama_vocab_type(const struct llama_model * model);
```
"""
function llama_vocab_type(model)
    ccall((:llama_vocab_type, libllama), llama_vocab_type, (Ptr{llama_model},), model)
end

"""
    llama_n_vocab(model)


### Prototype
```c
int32_t llama_n_vocab (const struct llama_model * model);
```
"""
function llama_n_vocab(model)
    ccall((:llama_n_vocab, libllama), Int32, (Ptr{llama_model},), model)
end

"""
    llama_n_ctx_train(model)


### Prototype
```c
int32_t llama_n_ctx_train(const struct llama_model * model);
```
"""
function llama_n_ctx_train(model)
    ccall((:llama_n_ctx_train, libllama), Int32, (Ptr{llama_model},), model)
end

"""
    llama_n_embd(model)


### Prototype
```c
int32_t llama_n_embd (const struct llama_model * model);
```
"""
function llama_n_embd(model)
    ccall((:llama_n_embd, libllama), Int32, (Ptr{llama_model},), model)
end

"""
    llama_rope_freq_scale_train(model)


### Prototype
```c
float llama_rope_freq_scale_train(const struct llama_model * model);
```
"""
function llama_rope_freq_scale_train(model)
    ccall((:llama_rope_freq_scale_train, libllama), Cfloat, (Ptr{llama_model},), model)
end

"""
    llama_model_meta_val_str(model, key, buf, buf_size)


### Prototype
```c
int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);
```
"""
function llama_model_meta_val_str(model, key, buf, buf_size)
    ccall((:llama_model_meta_val_str, libllama), Int32, (Ptr{llama_model}, Ptr{Cchar}, Ptr{Cchar}, Csize_t), model, key, buf, buf_size)
end

"""
    llama_model_meta_count(model)


### Prototype
```c
int32_t llama_model_meta_count(const struct llama_model * model);
```
"""
function llama_model_meta_count(model)
    ccall((:llama_model_meta_count, libllama), Int32, (Ptr{llama_model},), model)
end

"""
    llama_model_meta_key_by_index(model, i, buf, buf_size)


### Prototype
```c
int32_t llama_model_meta_key_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);
```
"""
function llama_model_meta_key_by_index(model, i, buf, buf_size)
    ccall((:llama_model_meta_key_by_index, libllama), Int32, (Ptr{llama_model}, Int32, Ptr{Cchar}, Csize_t), model, i, buf, buf_size)
end

"""
    llama_model_meta_val_str_by_index(model, i, buf, buf_size)


### Prototype
```c
int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);
```
"""
function llama_model_meta_val_str_by_index(model, i, buf, buf_size)
    ccall((:llama_model_meta_val_str_by_index, libllama), Int32, (Ptr{llama_model}, Int32, Ptr{Cchar}, Csize_t), model, i, buf, buf_size)
end

"""
    llama_model_desc(model, buf, buf_size)


### Prototype
```c
int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);
```
"""
function llama_model_desc(model, buf, buf_size)
    ccall((:llama_model_desc, libllama), Int32, (Ptr{llama_model}, Ptr{Cchar}, Csize_t), model, buf, buf_size)
end

"""
    llama_model_size(model)


### Prototype
```c
uint64_t llama_model_size(const struct llama_model * model);
```
"""
function llama_model_size(model)
    ccall((:llama_model_size, libllama), UInt64, (Ptr{llama_model},), model)
end

"""
    llama_model_n_params(model)


### Prototype
```c
uint64_t llama_model_n_params(const struct llama_model * model);
```
"""
function llama_model_n_params(model)
    ccall((:llama_model_n_params, libllama), UInt64, (Ptr{llama_model},), model)
end

@cenum ggml_backend_type::UInt32 begin
    GGML_BACKEND_CPU = 0
    GGML_BACKEND_GPU = 10
    GGML_BACKEND_GPU_SPLIT = 20
end

mutable struct ggml_backend_buffer end

@cenum ggml_op::UInt32 begin
    GGML_OP_NONE = 0
    GGML_OP_DUP = 1
    GGML_OP_ADD = 2
    GGML_OP_ADD1 = 3
    GGML_OP_ACC = 4
    GGML_OP_SUB = 5
    GGML_OP_MUL = 6
    GGML_OP_DIV = 7
    GGML_OP_SQR = 8
    GGML_OP_SQRT = 9
    GGML_OP_LOG = 10
    GGML_OP_SUM = 11
    GGML_OP_SUM_ROWS = 12
    GGML_OP_MEAN = 13
    GGML_OP_ARGMAX = 14
    GGML_OP_REPEAT = 15
    GGML_OP_REPEAT_BACK = 16
    GGML_OP_CONCAT = 17
    GGML_OP_SILU_BACK = 18
    GGML_OP_NORM = 19
    GGML_OP_RMS_NORM = 20
    GGML_OP_RMS_NORM_BACK = 21
    GGML_OP_GROUP_NORM = 22
    GGML_OP_MUL_MAT = 23
    GGML_OP_MUL_MAT_ID = 24
    GGML_OP_OUT_PROD = 25
    GGML_OP_SCALE = 26
    GGML_OP_SET = 27
    GGML_OP_CPY = 28
    GGML_OP_CONT = 29
    GGML_OP_RESHAPE = 30
    GGML_OP_VIEW = 31
    GGML_OP_PERMUTE = 32
    GGML_OP_TRANSPOSE = 33
    GGML_OP_GET_ROWS = 34
    GGML_OP_GET_ROWS_BACK = 35
    GGML_OP_DIAG = 36
    GGML_OP_DIAG_MASK_INF = 37
    GGML_OP_DIAG_MASK_ZERO = 38
    GGML_OP_SOFT_MAX = 39
    GGML_OP_SOFT_MAX_BACK = 40
    GGML_OP_ROPE = 41
    GGML_OP_ROPE_BACK = 42
    GGML_OP_ALIBI = 43
    GGML_OP_CLAMP = 44
    GGML_OP_CONV_TRANSPOSE_1D = 45
    GGML_OP_IM2COL = 46
    GGML_OP_CONV_TRANSPOSE_2D = 47
    GGML_OP_POOL_1D = 48
    GGML_OP_POOL_2D = 49
    GGML_OP_UPSCALE = 50
    GGML_OP_PAD = 51
    GGML_OP_ARGSORT = 52
    GGML_OP_LEAKY_RELU = 53
    GGML_OP_FLASH_ATTN = 54
    GGML_OP_FLASH_FF = 55
    GGML_OP_FLASH_ATTN_BACK = 56
    GGML_OP_WIN_PART = 57
    GGML_OP_WIN_UNPART = 58
    GGML_OP_GET_REL_POS = 59
    GGML_OP_ADD_REL_POS = 60
    GGML_OP_UNARY = 61
    GGML_OP_MAP_UNARY = 62
    GGML_OP_MAP_BINARY = 63
    GGML_OP_MAP_CUSTOM1_F32 = 64
    GGML_OP_MAP_CUSTOM2_F32 = 65
    GGML_OP_MAP_CUSTOM3_F32 = 66
    GGML_OP_MAP_CUSTOM1 = 67
    GGML_OP_MAP_CUSTOM2 = 68
    GGML_OP_MAP_CUSTOM3 = 69
    GGML_OP_CROSS_ENTROPY_LOSS = 70
    GGML_OP_CROSS_ENTROPY_LOSS_BACK = 71
    GGML_OP_COUNT = 72
end

struct ggml_tensor
    type::ggml_type
    backend::ggml_backend_type
    buffer::Ptr{ggml_backend_buffer}
    ne::NTuple{4, Int64} # ne::NTuple{4, Int64}
    nb::NTuple{4, Csize_t} # nb::NTuple{4, Csize_t}
    op::ggml_op
    op_params::NTuple{16, Int32} # op_params::NTuple{16, Int32}
    is_param::Bool
    grad::Ptr{ggml_tensor}
    src::NTuple{10, Ptr{ggml_tensor}}
    perf_runs::Cint
    perf_cycles::Int64
    perf_time_us::Int64
    view_src::Ptr{ggml_tensor}
    view_offs::Csize_t
    data::Ptr{Cvoid}
    name::NTuple{64, Cchar}
    extra::Ptr{Cvoid}
    padding::NTuple{8, Cchar}
end

function Base.getproperty(x::ggml_tensor, f::Symbol)
    f === :ne && return NTuple{4, Int64}(getfield(x, f))
    f === :nb && return NTuple{4, Csize_t}(getfield(x, f))
    f === :op_params && return NTuple{16, Int32}(getfield(x, f))
    return getfield(x, f)
end

"""
    llama_get_model_tensor(model, name)


### Prototype
```c
struct ggml_tensor * llama_get_model_tensor(struct llama_model * model, const char * name);
```
"""
function llama_get_model_tensor(model, name)
    ccall((:llama_get_model_tensor, libllama), Ptr{ggml_tensor}, (Ptr{llama_model}, Ptr{Cchar}), model, name)
end

"""
    llama_model_quantize(fname_inp, fname_out, params)


### Prototype
```c
uint32_t llama_model_quantize( const char * fname_inp, const char * fname_out, const llama_model_quantize_params * params);
```
"""
function llama_model_quantize(fname_inp, fname_out, params)
    ccall((:llama_model_quantize, libllama), UInt32, (Ptr{Cchar}, Ptr{Cchar}, Ptr{llama_model_quantize_params}), fname_inp, fname_out, params)
end

"""
    llama_apply_lora_from_file(ctx, path_lora, scale, path_base_model, n_threads)


### Prototype
```c
DEPRECATED(int32_t llama_apply_lora_from_file( struct llama_context * ctx, const char * path_lora, float scale, const char * path_base_model, int32_t n_threads), "use llama_model_apply_lora_from_file instead");
```
"""
function llama_apply_lora_from_file(ctx, path_lora, scale, path_base_model, n_threads)
    ccall((:llama_apply_lora_from_file, libllama), Int32, (Ptr{llama_context}, Ptr{Cchar}, Cfloat, Ptr{Cchar}, Int32), ctx, path_lora, scale, path_base_model, n_threads)
end

"""
    llama_model_apply_lora_from_file(model, path_lora, scale, path_base_model, n_threads)


### Prototype
```c
int32_t llama_model_apply_lora_from_file( const struct llama_model * model, const char * path_lora, float scale, const char * path_base_model, int32_t n_threads);
```
"""
function llama_model_apply_lora_from_file(model, path_lora, scale, path_base_model, n_threads)
    ccall((:llama_model_apply_lora_from_file, libllama), Int32, (Ptr{llama_model}, Ptr{Cchar}, Cfloat, Ptr{Cchar}, Int32), model, path_lora, scale, path_base_model, n_threads)
end

struct llama_kv_cache_view_cell
    pos::llama_pos
end

struct llama_kv_cache_view
    n_cells::Int32
    n_max_seq::Int32
    token_count::Int32
    used_cells::Int32
    max_contiguous::Int32
    max_contiguous_idx::Int32
    cells::Ptr{llama_kv_cache_view_cell}
    cells_sequences::Ptr{llama_seq_id}
end

"""
    llama_kv_cache_view_init(ctx, n_max_seq)


### Prototype
```c
struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_context * ctx, int32_t n_max_seq);
```
"""
function llama_kv_cache_view_init(ctx, n_max_seq)
    ccall((:llama_kv_cache_view_init, libllama), llama_kv_cache_view, (Ptr{llama_context}, Int32), ctx, n_max_seq)
end

"""
    llama_kv_cache_view_free(view)


### Prototype
```c
void llama_kv_cache_view_free(struct llama_kv_cache_view * view);
```
"""
function llama_kv_cache_view_free(view)
    ccall((:llama_kv_cache_view_free, libllama), Cvoid, (Ptr{llama_kv_cache_view},), view)
end

"""
    llama_kv_cache_view_update(ctx, view)


### Prototype
```c
void llama_kv_cache_view_update(const struct llama_context * ctx, struct llama_kv_cache_view * view);
```
"""
function llama_kv_cache_view_update(ctx, view)
    ccall((:llama_kv_cache_view_update, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_kv_cache_view}), ctx, view)
end

"""
    llama_get_kv_cache_token_count(ctx)


### Prototype
```c
int32_t llama_get_kv_cache_token_count(const struct llama_context * ctx);
```
"""
function llama_get_kv_cache_token_count(ctx)
    ccall((:llama_get_kv_cache_token_count, libllama), Int32, (Ptr{llama_context},), ctx)
end

"""
    llama_get_kv_cache_used_cells(ctx)


### Prototype
```c
int32_t llama_get_kv_cache_used_cells(const struct llama_context * ctx);
```
"""
function llama_get_kv_cache_used_cells(ctx)
    ccall((:llama_get_kv_cache_used_cells, libllama), Int32, (Ptr{llama_context},), ctx)
end

"""
    llama_kv_cache_clear(ctx)


### Prototype
```c
void llama_kv_cache_clear( struct llama_context * ctx);
```
"""
function llama_kv_cache_clear(ctx)
    ccall((:llama_kv_cache_clear, libllama), Cvoid, (Ptr{llama_context},), ctx)
end

"""
    llama_kv_cache_seq_rm(ctx, seq_id, p0, p1)


### Prototype
```c
void llama_kv_cache_seq_rm( struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
```
"""
function llama_kv_cache_seq_rm(ctx, seq_id, p0, p1)
    ccall((:llama_kv_cache_seq_rm, libllama), Cvoid, (Ptr{llama_context}, llama_seq_id, llama_pos, llama_pos), ctx, seq_id, p0, p1)
end

"""
    llama_kv_cache_seq_cp(ctx, seq_id_src, seq_id_dst, p0, p1)


### Prototype
```c
void llama_kv_cache_seq_cp( struct llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
```
"""
function llama_kv_cache_seq_cp(ctx, seq_id_src, seq_id_dst, p0, p1)
    ccall((:llama_kv_cache_seq_cp, libllama), Cvoid, (Ptr{llama_context}, llama_seq_id, llama_seq_id, llama_pos, llama_pos), ctx, seq_id_src, seq_id_dst, p0, p1)
end

"""
    llama_kv_cache_seq_keep(ctx, seq_id)


### Prototype
```c
void llama_kv_cache_seq_keep( struct llama_context * ctx, llama_seq_id seq_id);
```
"""
function llama_kv_cache_seq_keep(ctx, seq_id)
    ccall((:llama_kv_cache_seq_keep, libllama), Cvoid, (Ptr{llama_context}, llama_seq_id), ctx, seq_id)
end

"""
    llama_kv_cache_seq_shift(ctx, seq_id, p0, p1, delta)


### Prototype
```c
void llama_kv_cache_seq_shift( struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta);
```
"""
function llama_kv_cache_seq_shift(ctx, seq_id, p0, p1, delta)
    ccall((:llama_kv_cache_seq_shift, libllama), Cvoid, (Ptr{llama_context}, llama_seq_id, llama_pos, llama_pos, llama_pos), ctx, seq_id, p0, p1, delta)
end

"""
    llama_kv_cache_seq_div(ctx, seq_id, p0, p1, d)


### Prototype
```c
void llama_kv_cache_seq_div( struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d);
```
"""
function llama_kv_cache_seq_div(ctx, seq_id, p0, p1, d)
    ccall((:llama_kv_cache_seq_div, libllama), Cvoid, (Ptr{llama_context}, llama_seq_id, llama_pos, llama_pos, Cint), ctx, seq_id, p0, p1, d)
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
size_t llama_copy_state_data( struct llama_context * ctx, uint8_t * dst);
```
"""
function llama_copy_state_data(ctx, dst)
    ccall((:llama_copy_state_data, libllama), Csize_t, (Ptr{llama_context}, Ptr{UInt8}), ctx, dst)
end

"""
    llama_set_state_data(ctx, src)


### Prototype
```c
size_t llama_set_state_data( struct llama_context * ctx, uint8_t * src);
```
"""
function llama_set_state_data(ctx, src)
    ccall((:llama_set_state_data, libllama), Csize_t, (Ptr{llama_context}, Ptr{UInt8}), ctx, src)
end

"""
    llama_load_session_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)


### Prototype
```c
bool llama_load_session_file( struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
```
"""
function llama_load_session_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)
    ccall((:llama_load_session_file, libllama), Bool, (Ptr{llama_context}, Ptr{Cchar}, Ptr{llama_token}, Csize_t, Ptr{Csize_t}), ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)
end

"""
    llama_save_session_file(ctx, path_session, tokens, n_token_count)


### Prototype
```c
bool llama_save_session_file( struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count);
```
"""
function llama_save_session_file(ctx, path_session, tokens, n_token_count)
    ccall((:llama_save_session_file, libllama), Bool, (Ptr{llama_context}, Ptr{Cchar}, Ptr{llama_token}, Csize_t), ctx, path_session, tokens, n_token_count)
end

"""
    llama_eval(ctx, tokens, n_tokens, n_past)


### Prototype
```c
DEPRECATED(int llama_eval( struct llama_context * ctx, llama_token * tokens, int32_t n_tokens, int32_t n_past), "use llama_decode() instead");
```
"""
function llama_eval(ctx, tokens, n_tokens, n_past)
    ccall((:llama_eval, libllama), Cint, (Ptr{llama_context}, Ptr{llama_token}, Int32, Int32), ctx, tokens, n_tokens, n_past)
end

"""
    llama_eval_embd(ctx, embd, n_tokens, n_past)


### Prototype
```c
DEPRECATED(int llama_eval_embd( struct llama_context * ctx, float * embd, int32_t n_tokens, int32_t n_past), "use llama_decode() instead");
```
"""
function llama_eval_embd(ctx, embd, n_tokens, n_past)
    ccall((:llama_eval_embd, libllama), Cint, (Ptr{llama_context}, Ptr{Cfloat}, Int32, Int32), ctx, embd, n_tokens, n_past)
end

"""
    llama_batch_get_one(tokens, n_tokens, pos_0, seq_id)


### Prototype
```c
struct llama_batch llama_batch_get_one( llama_token * tokens, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id);
```
"""
function llama_batch_get_one(tokens, n_tokens, pos_0, seq_id)
    ccall((:llama_batch_get_one, libllama), llama_batch, (Ptr{llama_token}, Int32, llama_pos, llama_seq_id), tokens, n_tokens, pos_0, seq_id)
end

"""
    llama_batch_init(n_tokens, embd, n_seq_max)


### Prototype
```c
struct llama_batch llama_batch_init( int32_t n_tokens, int32_t embd, int32_t n_seq_max);
```
"""
function llama_batch_init(n_tokens, embd, n_seq_max)
    ccall((:llama_batch_init, libllama), llama_batch, (Int32, Int32, Int32), n_tokens, embd, n_seq_max)
end

"""
    llama_batch_free(batch)


### Prototype
```c
void llama_batch_free(struct llama_batch batch);
```
"""
function llama_batch_free(batch)
    ccall((:llama_batch_free, libllama), Cvoid, (llama_batch,), batch)
end

"""
    llama_decode(ctx, batch)


### Prototype
```c
int32_t llama_decode( struct llama_context * ctx, struct llama_batch batch);
```
"""
function llama_decode(ctx, batch)
    ccall((:llama_decode, libllama), Int32, (Ptr{llama_context}, llama_batch), ctx, batch)
end

"""
    llama_set_n_threads(ctx, n_threads, n_threads_batch)


### Prototype
```c
void llama_set_n_threads(struct llama_context * ctx, uint32_t n_threads, uint32_t n_threads_batch);
```
"""
function llama_set_n_threads(ctx, n_threads, n_threads_batch)
    ccall((:llama_set_n_threads, libllama), Cvoid, (Ptr{llama_context}, UInt32, UInt32), ctx, n_threads, n_threads_batch)
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
    llama_get_logits_ith(ctx, i)


### Prototype
```c
float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);
```
"""
function llama_get_logits_ith(ctx, i)
    ccall((:llama_get_logits_ith, libllama), Ptr{Cfloat}, (Ptr{llama_context}, Int32), ctx, i)
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
    llama_token_get_text(model, token)


### Prototype
```c
const char * llama_token_get_text(const struct llama_model * model, llama_token token);
```
"""
function llama_token_get_text(model, token)
    ccall((:llama_token_get_text, libllama), Ptr{Cchar}, (Ptr{llama_model}, llama_token), model, token)
end

"""
    llama_token_get_score(model, token)


### Prototype
```c
float llama_token_get_score(const struct llama_model * model, llama_token token);
```
"""
function llama_token_get_score(model, token)
    ccall((:llama_token_get_score, libllama), Cfloat, (Ptr{llama_model}, llama_token), model, token)
end

"""
    llama_token_get_type(model, token)


### Prototype
```c
enum llama_token_type llama_token_get_type(const struct llama_model * model, llama_token token);
```
"""
function llama_token_get_type(model, token)
    ccall((:llama_token_get_type, libllama), llama_token_type, (Ptr{llama_model}, llama_token), model, token)
end

"""
    llama_token_bos(model)


### Prototype
```c
llama_token llama_token_bos(const struct llama_model * model);
```
"""
function llama_token_bos(model)
    ccall((:llama_token_bos, libllama), llama_token, (Ptr{llama_model},), model)
end

"""
    llama_token_eos(model)


### Prototype
```c
llama_token llama_token_eos(const struct llama_model * model);
```
"""
function llama_token_eos(model)
    ccall((:llama_token_eos, libllama), llama_token, (Ptr{llama_model},), model)
end

"""
    llama_token_nl(model)


### Prototype
```c
llama_token llama_token_nl (const struct llama_model * model);
```
"""
function llama_token_nl(model)
    ccall((:llama_token_nl, libllama), llama_token, (Ptr{llama_model},), model)
end

"""
    llama_add_bos_token(model)


### Prototype
```c
int32_t llama_add_bos_token(const struct llama_model * model);
```
"""
function llama_add_bos_token(model)
    ccall((:llama_add_bos_token, libllama), Int32, (Ptr{llama_model},), model)
end

"""
    llama_add_eos_token(model)


### Prototype
```c
int32_t llama_add_eos_token(const struct llama_model * model);
```
"""
function llama_add_eos_token(model)
    ccall((:llama_add_eos_token, libllama), Int32, (Ptr{llama_model},), model)
end

"""
    llama_token_prefix(model)


### Prototype
```c
llama_token llama_token_prefix(const struct llama_model * model);
```
"""
function llama_token_prefix(model)
    ccall((:llama_token_prefix, libllama), llama_token, (Ptr{llama_model},), model)
end

"""
    llama_token_middle(model)


### Prototype
```c
llama_token llama_token_middle(const struct llama_model * model);
```
"""
function llama_token_middle(model)
    ccall((:llama_token_middle, libllama), llama_token, (Ptr{llama_model},), model)
end

"""
    llama_token_suffix(model)


### Prototype
```c
llama_token llama_token_suffix(const struct llama_model * model);
```
"""
function llama_token_suffix(model)
    ccall((:llama_token_suffix, libllama), llama_token, (Ptr{llama_model},), model)
end

"""
    llama_token_eot(model)


### Prototype
```c
llama_token llama_token_eot (const struct llama_model * model);
```
"""
function llama_token_eot(model)
    ccall((:llama_token_eot, libllama), llama_token, (Ptr{llama_model},), model)
end

"""
    llama_tokenize(model, text, text_len, tokens, n_max_tokens, add_bos, special)

@details Convert the provided text into tokens.
@param tokens The tokens pointer must be large enough to hold the resulting tokens.
@return Returns the number of tokens on success, no more than n_max_tokens
@return Returns a negative number on failure - the number of tokens that would have been returned
@param special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext.
Does not insert a leading space.
### Prototype
```c
int32_t llama_tokenize( const struct llama_model * model, const char * text, int32_t text_len, llama_token * tokens, int32_t n_max_tokens, bool add_bos, bool special);
```
"""
function llama_tokenize(model, text, text_len, tokens, n_max_tokens, add_bos, special)
    ccall((:llama_tokenize, libllama), Int32, (Ptr{llama_model}, Ptr{Cchar}, Int32, Ptr{llama_token}, Int32, Bool, Bool), model, text, text_len, tokens, n_max_tokens, add_bos, special)
end

"""
    llama_token_to_piece(model, token, buf, length)


### Prototype
```c
int32_t llama_token_to_piece( const struct llama_model * model, llama_token token, char * buf, int32_t length);
```
"""
function llama_token_to_piece(model, token, buf, length)
    ccall((:llama_token_to_piece, libllama), Int32, (Ptr{llama_model}, llama_token, Ptr{Cchar}, Int32), model, token, buf, length)
end

"""
    llama_grammar_init(rules, n_rules, start_rule_index)


### Prototype
```c
struct llama_grammar * llama_grammar_init( const llama_grammar_element ** rules, size_t n_rules, size_t start_rule_index);
```
"""
function llama_grammar_init(rules, n_rules, start_rule_index)
    ccall((:llama_grammar_init, libllama), Ptr{llama_grammar}, (Ptr{Ptr{llama_grammar_element}}, Csize_t, Csize_t), rules, n_rules, start_rule_index)
end

"""
    llama_grammar_free(grammar)


### Prototype
```c
void llama_grammar_free(struct llama_grammar * grammar);
```
"""
function llama_grammar_free(grammar)
    ccall((:llama_grammar_free, libllama), Cvoid, (Ptr{llama_grammar},), grammar)
end

"""
    llama_grammar_copy(grammar)


### Prototype
```c
struct llama_grammar * llama_grammar_copy(const struct llama_grammar * grammar);
```
"""
function llama_grammar_copy(grammar)
    ccall((:llama_grammar_copy, libllama), Ptr{llama_grammar}, (Ptr{llama_grammar},), grammar)
end

"""
    llama_set_rng_seed(ctx, seed)


### Prototype
```c
void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed);
```
"""
function llama_set_rng_seed(ctx, seed)
    ccall((:llama_set_rng_seed, libllama), Cvoid, (Ptr{llama_context}, UInt32), ctx, seed)
end

"""
    llama_sample_repetition_penalties(ctx, candidates, last_tokens, penalty_last_n, penalty_repeat, penalty_freq, penalty_present)

@details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
@details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
### Prototype
```c
void llama_sample_repetition_penalties( struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present);
```
"""
function llama_sample_repetition_penalties(ctx, candidates, last_tokens, penalty_last_n, penalty_repeat, penalty_freq, penalty_present)
    ccall((:llama_sample_repetition_penalties, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Ptr{llama_token}, Csize_t, Cfloat, Cfloat, Cfloat), ctx, candidates, last_tokens, penalty_last_n, penalty_repeat, penalty_freq, penalty_present)
end

"""
    llama_sample_classifier_free_guidance(ctx, candidates, guidance_ctx, scale)

@details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
@param candidates A vector of `llama_token_data` containing the candidate tokens, the logits must be directly extracted from the original generation context without being sorted.
@params guidance_ctx A separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
@params scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
### Prototype
```c
void llama_sample_classifier_free_guidance( struct llama_context * ctx, llama_token_data_array * candidates, struct llama_context * guidance_ctx, float scale);
```
"""
function llama_sample_classifier_free_guidance(ctx, candidates, guidance_ctx, scale)
    ccall((:llama_sample_classifier_free_guidance, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Ptr{llama_context}, Cfloat), ctx, candidates, guidance_ctx, scale)
end

"""
    llama_sample_softmax(ctx, candidates)

@details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
### Prototype
```c
void llama_sample_softmax( struct llama_context * ctx, llama_token_data_array * candidates);
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
void llama_sample_top_k( struct llama_context * ctx, llama_token_data_array * candidates, int32_t k, size_t min_keep);
```
"""
function llama_sample_top_k(ctx, candidates, k, min_keep)
    ccall((:llama_sample_top_k, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Int32, Csize_t), ctx, candidates, k, min_keep)
end

"""
    llama_sample_top_p(ctx, candidates, p, min_keep)

@details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
### Prototype
```c
void llama_sample_top_p( struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);
```
"""
function llama_sample_top_p(ctx, candidates, p, min_keep)
    ccall((:llama_sample_top_p, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat, Csize_t), ctx, candidates, p, min_keep)
end

"""
    llama_sample_min_p(ctx, candidates, p, min_keep)

@details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
### Prototype
```c
void llama_sample_min_p( struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);
```
"""
function llama_sample_min_p(ctx, candidates, p, min_keep)
    ccall((:llama_sample_min_p, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat, Csize_t), ctx, candidates, p, min_keep)
end

"""
    llama_sample_tail_free(ctx, candidates, z, min_keep)

@details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
### Prototype
```c
void llama_sample_tail_free( struct llama_context * ctx, llama_token_data_array * candidates, float z, size_t min_keep);
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
void llama_sample_typical( struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);
```
"""
function llama_sample_typical(ctx, candidates, p, min_keep)
    ccall((:llama_sample_typical, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat, Csize_t), ctx, candidates, p, min_keep)
end

"""
    llama_sample_temp(ctx, candidates, temp)


### Prototype
```c
void llama_sample_temp( struct llama_context * ctx, llama_token_data_array * candidates, float temp);
```
"""
function llama_sample_temp(ctx, candidates, temp)
    ccall((:llama_sample_temp, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat), ctx, candidates, temp)
end

"""
    llama_sample_temperature(ctx, candidates, temp)


### Prototype
```c
DEPRECATED(void llama_sample_temperature( struct llama_context * ctx, llama_token_data_array * candidates, float temp), "use llama_sample_temp instead");
```
"""
function llama_sample_temperature(ctx, candidates, temp)
    ccall((:llama_sample_temperature, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat), ctx, candidates, temp)
end

"""
    llama_sample_grammar(ctx, candidates, grammar)

@details Apply constraints from grammar
### Prototype
```c
void llama_sample_grammar( struct llama_context * ctx, llama_token_data_array * candidates, const struct llama_grammar * grammar);
```
"""
function llama_sample_grammar(ctx, candidates, grammar)
    ccall((:llama_sample_grammar, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_token_data_array}, Ptr{llama_grammar}), ctx, candidates, grammar)
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
llama_token llama_sample_token_mirostat( struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, int32_t m, float * mu);
```
"""
function llama_sample_token_mirostat(ctx, candidates, tau, eta, m, mu)
    ccall((:llama_sample_token_mirostat, libllama), llama_token, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat, Cfloat, Int32, Ptr{Cfloat}), ctx, candidates, tau, eta, m, mu)
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
llama_token llama_sample_token_mirostat_v2( struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, float * mu);
```
"""
function llama_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu)
    ccall((:llama_sample_token_mirostat_v2, libllama), llama_token, (Ptr{llama_context}, Ptr{llama_token_data_array}, Cfloat, Cfloat, Ptr{Cfloat}), ctx, candidates, tau, eta, mu)
end

"""
    llama_sample_token_greedy(ctx, candidates)

@details Selects the token with the highest probability.
Does not compute the token probabilities. Use llama_sample_softmax() instead.
### Prototype
```c
llama_token llama_sample_token_greedy( struct llama_context * ctx, llama_token_data_array * candidates);
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
llama_token llama_sample_token( struct llama_context * ctx, llama_token_data_array * candidates);
```
"""
function llama_sample_token(ctx, candidates)
    ccall((:llama_sample_token, libllama), llama_token, (Ptr{llama_context}, Ptr{llama_token_data_array}), ctx, candidates)
end

"""
    llama_grammar_accept_token(ctx, grammar, token)

@details Accepts the sampled token into the grammar
### Prototype
```c
void llama_grammar_accept_token( struct llama_context * ctx, struct llama_grammar * grammar, llama_token token);
```
"""
function llama_grammar_accept_token(ctx, grammar, token)
    ccall((:llama_grammar_accept_token, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_grammar}, llama_token), ctx, grammar, token)
end

struct llama_beam_view
    tokens::Ptr{llama_token}
    n_tokens::Csize_t
    p::Cfloat
    eob::Bool
end

struct llama_beams_state
    beam_views::Ptr{llama_beam_view}
    n_beams::Csize_t
    common_prefix_length::Csize_t
    last_call::Bool
end

# typedef void ( * llama_beam_search_callback_fn_t ) ( void * callback_data , struct llama_beams_state )
const llama_beam_search_callback_fn_t = Ptr{Cvoid}

"""
    llama_beam_search(ctx, callback, callback_data, n_beams, n_past, n_predict)

@details Deterministically returns entire sentence constructed by a beam search.
@param ctx Pointer to the llama_context.
@param callback Invoked for each iteration of the beam_search loop, passing in beams_state.
@param callback_data A pointer that is simply passed back to callback.
@param n_beams Number of beams to use.
@param n_past Number of tokens already evaluated.
@param n_predict Maximum number of tokens to predict. EOS may occur earlier.
### Prototype
```c
void llama_beam_search( struct llama_context * ctx, llama_beam_search_callback_fn_t callback, void * callback_data, size_t n_beams, int32_t n_past, int32_t n_predict);
```
"""
function llama_beam_search(ctx, callback, callback_data, n_beams, n_past, n_predict)
    ccall((:llama_beam_search, libllama), Cvoid, (Ptr{llama_context}, llama_beam_search_callback_fn_t, Ptr{Cvoid}, Csize_t, Int32, Int32), ctx, callback, callback_data, n_beams, n_past, n_predict)
end

"""
    llama_get_timings(ctx)


### Prototype
```c
struct llama_timings llama_get_timings(struct llama_context * ctx);
```
"""
function llama_get_timings(ctx)
    ccall((:llama_get_timings, libllama), llama_timings, (Ptr{llama_context},), ctx)
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

# typedef void ( * ggml_log_callback ) ( enum ggml_log_level level , const char * text , void * user_data )
const ggml_log_callback = Ptr{Cvoid}

"""
    llama_log_set(log_callback, user_data)


### Prototype
```c
void llama_log_set(ggml_log_callback log_callback, void * user_data);
```
"""
function llama_log_set(log_callback, user_data)
    ccall((:llama_log_set, libllama), Cvoid, (ggml_log_callback, Ptr{Cvoid}), log_callback, user_data)
end

"""
    llama_dump_timing_info_yaml(stream, ctx)


### Prototype
```c
void llama_dump_timing_info_yaml(FILE * stream, const struct llama_context * ctx);
```
"""
function llama_dump_timing_info_yaml(stream, ctx)
    ccall((:llama_dump_timing_info_yaml, libllama), Cvoid, (Ptr{Libc.FILE}, Ptr{llama_context}), stream, ctx)
end

"""
    ggml_print_backtrace()


### Prototype
```c
void ggml_print_backtrace(void);
```
"""
function ggml_print_backtrace()
    ccall((:ggml_print_backtrace, libllama), Cvoid, ())
end

const ggml_fp16_t = UInt16

"""
    ggml_fp16_to_fp32(x)


### Prototype
```c
float ggml_fp16_to_fp32(ggml_fp16_t x);
```
"""
function ggml_fp16_to_fp32(x)
    ccall((:ggml_fp16_to_fp32, libllama), Cfloat, (ggml_fp16_t,), x)
end

"""
    ggml_fp32_to_fp16(x)


### Prototype
```c
ggml_fp16_t ggml_fp32_to_fp16(float x);
```
"""
function ggml_fp32_to_fp16(x)
    ccall((:ggml_fp32_to_fp16, libllama), ggml_fp16_t, (Cfloat,), x)
end

"""
    ggml_fp16_to_fp32_row(x, y, n)


### Prototype
```c
void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int n);
```
"""
function ggml_fp16_to_fp32_row(x, y, n)
    ccall((:ggml_fp16_to_fp32_row, libllama), Cvoid, (Ptr{ggml_fp16_t}, Ptr{Cfloat}, Cint), x, y, n)
end

"""
    ggml_fp32_to_fp16_row(x, y, n)


### Prototype
```c
void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int n);
```
"""
function ggml_fp32_to_fp16_row(x, y, n)
    ccall((:ggml_fp32_to_fp16_row, libllama), Cvoid, (Ptr{Cfloat}, Ptr{ggml_fp16_t}, Cint), x, y, n)
end

@cenum ggml_prec::UInt32 begin
    GGML_PREC_DEFAULT = 0
    GGML_PREC_F32 = 1
end

@cenum ggml_ftype::Int32 begin
    GGML_FTYPE_UNKNOWN = -1
    GGML_FTYPE_ALL_F32 = 0
    GGML_FTYPE_MOSTLY_F16 = 1
    GGML_FTYPE_MOSTLY_Q4_0 = 2
    GGML_FTYPE_MOSTLY_Q4_1 = 3
    GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
    GGML_FTYPE_MOSTLY_Q8_0 = 7
    GGML_FTYPE_MOSTLY_Q5_0 = 8
    GGML_FTYPE_MOSTLY_Q5_1 = 9
    GGML_FTYPE_MOSTLY_Q2_K = 10
    GGML_FTYPE_MOSTLY_Q3_K = 11
    GGML_FTYPE_MOSTLY_Q4_K = 12
    GGML_FTYPE_MOSTLY_Q5_K = 13
    GGML_FTYPE_MOSTLY_Q6_K = 14
    GGML_FTYPE_MOSTLY_IQ2_XXS = 15
end

@cenum ggml_unary_op::UInt32 begin
    GGML_UNARY_OP_ABS = 0
    GGML_UNARY_OP_SGN = 1
    GGML_UNARY_OP_NEG = 2
    GGML_UNARY_OP_STEP = 3
    GGML_UNARY_OP_TANH = 4
    GGML_UNARY_OP_ELU = 5
    GGML_UNARY_OP_RELU = 6
    GGML_UNARY_OP_GELU = 7
    GGML_UNARY_OP_GELU_QUICK = 8
    GGML_UNARY_OP_SILU = 9
    GGML_UNARY_OP_COUNT = 10
end

@cenum ggml_object_type::UInt32 begin
    GGML_OBJECT_TENSOR = 0
    GGML_OBJECT_GRAPH = 1
    GGML_OBJECT_WORK_BUFFER = 2
end

@cenum ggml_log_level::UInt32 begin
    GGML_LOG_LEVEL_ERROR = 2
    GGML_LOG_LEVEL_WARN = 3
    GGML_LOG_LEVEL_INFO = 4
    GGML_LOG_LEVEL_DEBUG = 5
end

struct ggml_object
    offs::Csize_t
    size::Csize_t
    next::Ptr{ggml_object}
    type::ggml_object_type
    padding::NTuple{4, Cchar}
end

struct ggml_cplan
    work_size::Csize_t
    work_data::Ptr{UInt8}
    n_threads::Cint
    abort_callback::Ptr{Cvoid}
    abort_callback_data::Ptr{Cvoid}
end

@cenum ggml_cgraph_eval_order::UInt32 begin
    GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0
    GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT = 1
    GGML_CGRAPH_EVAL_ORDER_COUNT = 2
end

struct ggml_hash_set
    size::Csize_t
    keys::Ptr{Ptr{ggml_tensor}}
end

struct ggml_cgraph
    size::Cint
    n_nodes::Cint
    n_leafs::Cint
    nodes::Ptr{Ptr{ggml_tensor}}
    grads::Ptr{Ptr{ggml_tensor}}
    leafs::Ptr{Ptr{ggml_tensor}}
    visited_hash_table::ggml_hash_set
    order::ggml_cgraph_eval_order
    perf_runs::Cint
    perf_cycles::Int64
    perf_time_us::Int64
end

struct ggml_scratch
    offs::Csize_t
    size::Csize_t
    data::Ptr{Cvoid}
end

struct ggml_init_params
    mem_size::Csize_t
    mem_buffer::Ptr{Cvoid}
    no_alloc::Bool
end

@cenum ggml_task_type::UInt32 begin
    GGML_TASK_INIT = 0
    GGML_TASK_COMPUTE = 1
    GGML_TASK_FINALIZE = 2
end

struct ggml_compute_params
    type::ggml_task_type
    ith::Cint
    nth::Cint
    wsize::Csize_t
    wdata::Ptr{Cvoid}
end

"""
    ggml_time_init()


### Prototype
```c
void ggml_time_init(void);
```
"""
function ggml_time_init()
    ccall((:ggml_time_init, libllama), Cvoid, ())
end

"""
    ggml_time_ms()


### Prototype
```c
int64_t ggml_time_ms(void);
```
"""
function ggml_time_ms()
    ccall((:ggml_time_ms, libllama), Int64, ())
end

"""
    ggml_time_us()


### Prototype
```c
int64_t ggml_time_us(void);
```
"""
function ggml_time_us()
    ccall((:ggml_time_us, libllama), Int64, ())
end

"""
    ggml_cycles()


### Prototype
```c
int64_t ggml_cycles(void);
```
"""
function ggml_cycles()
    ccall((:ggml_cycles, libllama), Int64, ())
end

"""
    ggml_cycles_per_ms()


### Prototype
```c
int64_t ggml_cycles_per_ms(void);
```
"""
function ggml_cycles_per_ms()
    ccall((:ggml_cycles_per_ms, libllama), Int64, ())
end

"""
    ggml_numa_init()


### Prototype
```c
void ggml_numa_init(void);
```
"""
function ggml_numa_init()
    ccall((:ggml_numa_init, libllama), Cvoid, ())
end

"""
    ggml_is_numa()


### Prototype
```c
bool ggml_is_numa(void);
```
"""
function ggml_is_numa()
    ccall((:ggml_is_numa, libllama), Bool, ())
end

"""
    ggml_print_object(obj)


### Prototype
```c
void ggml_print_object (const struct ggml_object * obj);
```
"""
function ggml_print_object(obj)
    ccall((:ggml_print_object, libllama), Cvoid, (Ptr{ggml_object},), obj)
end

mutable struct ggml_context end

"""
    ggml_print_objects(ctx)


### Prototype
```c
void ggml_print_objects(const struct ggml_context * ctx);
```
"""
function ggml_print_objects(ctx)
    ccall((:ggml_print_objects, libllama), Cvoid, (Ptr{ggml_context},), ctx)
end

"""
    ggml_nelements(tensor)


### Prototype
```c
int64_t ggml_nelements (const struct ggml_tensor * tensor);
```
"""
function ggml_nelements(tensor)
    ccall((:ggml_nelements, libllama), Int64, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_nrows(tensor)


### Prototype
```c
int64_t ggml_nrows (const struct ggml_tensor * tensor);
```
"""
function ggml_nrows(tensor)
    ccall((:ggml_nrows, libllama), Int64, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_nbytes(tensor)


### Prototype
```c
size_t ggml_nbytes (const struct ggml_tensor * tensor);
```
"""
function ggml_nbytes(tensor)
    ccall((:ggml_nbytes, libllama), Csize_t, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_nbytes_pad(tensor)


### Prototype
```c
size_t ggml_nbytes_pad (const struct ggml_tensor * tensor);
```
"""
function ggml_nbytes_pad(tensor)
    ccall((:ggml_nbytes_pad, libllama), Csize_t, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_blck_size(type)


### Prototype
```c
int ggml_blck_size(enum ggml_type type);
```
"""
function ggml_blck_size(type)
    ccall((:ggml_blck_size, libllama), Cint, (ggml_type,), type)
end

"""
    ggml_type_size(type)


### Prototype
```c
size_t ggml_type_size(enum ggml_type type);
```
"""
function ggml_type_size(type)
    ccall((:ggml_type_size, libllama), Csize_t, (ggml_type,), type)
end

"""
    ggml_row_size(type, ne)


### Prototype
```c
size_t ggml_row_size (enum ggml_type type, int64_t ne);
```
"""
function ggml_row_size(type, ne)
    ccall((:ggml_row_size, libllama), Csize_t, (ggml_type, Int64), type, ne)
end

"""
    ggml_type_sizef(type)


### Prototype
```c
GGML_DEPRECATED( GGML_API double ggml_type_sizef(enum ggml_type type), "use ggml_row_size() instead");
```
"""
function ggml_type_sizef(type)
    ccall((:ggml_type_sizef, libllama), Cdouble, (ggml_type,), type)
end

"""
    ggml_type_name(type)


### Prototype
```c
const char * ggml_type_name(enum ggml_type type);
```
"""
function ggml_type_name(type)
    ccall((:ggml_type_name, libllama), Ptr{Cchar}, (ggml_type,), type)
end

"""
    ggml_op_name(op)


### Prototype
```c
const char * ggml_op_name (enum ggml_op op);
```
"""
function ggml_op_name(op)
    ccall((:ggml_op_name, libllama), Ptr{Cchar}, (ggml_op,), op)
end

"""
    ggml_op_symbol(op)


### Prototype
```c
const char * ggml_op_symbol(enum ggml_op op);
```
"""
function ggml_op_symbol(op)
    ccall((:ggml_op_symbol, libllama), Ptr{Cchar}, (ggml_op,), op)
end

"""
    ggml_unary_op_name(op)


### Prototype
```c
const char * ggml_unary_op_name(enum ggml_unary_op op);
```
"""
function ggml_unary_op_name(op)
    ccall((:ggml_unary_op_name, libllama), Ptr{Cchar}, (ggml_unary_op,), op)
end

"""
    ggml_op_desc(t)


### Prototype
```c
const char * ggml_op_desc(const struct ggml_tensor * t);
```
"""
function ggml_op_desc(t)
    ccall((:ggml_op_desc, libllama), Ptr{Cchar}, (Ptr{ggml_tensor},), t)
end

"""
    ggml_element_size(tensor)


### Prototype
```c
size_t ggml_element_size(const struct ggml_tensor * tensor);
```
"""
function ggml_element_size(tensor)
    ccall((:ggml_element_size, libllama), Csize_t, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_is_quantized(type)


### Prototype
```c
bool ggml_is_quantized(enum ggml_type type);
```
"""
function ggml_is_quantized(type)
    ccall((:ggml_is_quantized, libllama), Bool, (ggml_type,), type)
end

"""
    ggml_ftype_to_ggml_type(ftype)


### Prototype
```c
enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
```
"""
function ggml_ftype_to_ggml_type(ftype)
    ccall((:ggml_ftype_to_ggml_type, libllama), ggml_type, (ggml_ftype,), ftype)
end

"""
    ggml_is_transposed(tensor)


### Prototype
```c
bool ggml_is_transposed(const struct ggml_tensor * tensor);
```
"""
function ggml_is_transposed(tensor)
    ccall((:ggml_is_transposed, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_is_contiguous(tensor)


### Prototype
```c
bool ggml_is_contiguous(const struct ggml_tensor * tensor);
```
"""
function ggml_is_contiguous(tensor)
    ccall((:ggml_is_contiguous, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_is_permuted(tensor)


### Prototype
```c
bool ggml_is_permuted (const struct ggml_tensor * tensor);
```
"""
function ggml_is_permuted(tensor)
    ccall((:ggml_is_permuted, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_is_scalar(tensor)


### Prototype
```c
bool ggml_is_scalar (const struct ggml_tensor * tensor);
```
"""
function ggml_is_scalar(tensor)
    ccall((:ggml_is_scalar, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_is_vector(tensor)


### Prototype
```c
bool ggml_is_vector (const struct ggml_tensor * tensor);
```
"""
function ggml_is_vector(tensor)
    ccall((:ggml_is_vector, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_is_matrix(tensor)


### Prototype
```c
bool ggml_is_matrix (const struct ggml_tensor * tensor);
```
"""
function ggml_is_matrix(tensor)
    ccall((:ggml_is_matrix, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_is_3d(tensor)


### Prototype
```c
bool ggml_is_3d (const struct ggml_tensor * tensor);
```
"""
function ggml_is_3d(tensor)
    ccall((:ggml_is_3d, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_n_dims(tensor)


### Prototype
```c
int ggml_n_dims (const struct ggml_tensor * tensor);
```
"""
function ggml_n_dims(tensor)
    ccall((:ggml_n_dims, libllama), Cint, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_are_same_shape(t0, t1)


### Prototype
```c
bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
```
"""
function ggml_are_same_shape(t0, t1)
    ccall((:ggml_are_same_shape, libllama), Bool, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), t0, t1)
end

"""
    ggml_tensor_overhead()


### Prototype
```c
size_t ggml_tensor_overhead(void);
```
"""
function ggml_tensor_overhead()
    ccall((:ggml_tensor_overhead, libllama), Csize_t, ())
end

"""
    ggml_init(params)


### Prototype
```c
struct ggml_context * ggml_init(struct ggml_init_params params);
```
"""
function ggml_init(params)
    ccall((:ggml_init, libllama), Ptr{ggml_context}, (ggml_init_params,), params)
end

"""
    ggml_free(ctx)


### Prototype
```c
void ggml_free(struct ggml_context * ctx);
```
"""
function ggml_free(ctx)
    ccall((:ggml_free, libllama), Cvoid, (Ptr{ggml_context},), ctx)
end

"""
    ggml_used_mem(ctx)


### Prototype
```c
size_t ggml_used_mem(const struct ggml_context * ctx);
```
"""
function ggml_used_mem(ctx)
    ccall((:ggml_used_mem, libllama), Csize_t, (Ptr{ggml_context},), ctx)
end

"""
    ggml_set_scratch(ctx, scratch)


### Prototype
```c
size_t ggml_set_scratch (struct ggml_context * ctx, struct ggml_scratch scratch);
```
"""
function ggml_set_scratch(ctx, scratch)
    ccall((:ggml_set_scratch, libllama), Csize_t, (Ptr{ggml_context}, ggml_scratch), ctx, scratch)
end

"""
    ggml_get_no_alloc(ctx)


### Prototype
```c
bool ggml_get_no_alloc(struct ggml_context * ctx);
```
"""
function ggml_get_no_alloc(ctx)
    ccall((:ggml_get_no_alloc, libllama), Bool, (Ptr{ggml_context},), ctx)
end

"""
    ggml_set_no_alloc(ctx, no_alloc)


### Prototype
```c
void ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc);
```
"""
function ggml_set_no_alloc(ctx, no_alloc)
    ccall((:ggml_set_no_alloc, libllama), Cvoid, (Ptr{ggml_context}, Bool), ctx, no_alloc)
end

"""
    ggml_get_mem_buffer(ctx)


### Prototype
```c
void * ggml_get_mem_buffer (const struct ggml_context * ctx);
```
"""
function ggml_get_mem_buffer(ctx)
    ccall((:ggml_get_mem_buffer, libllama), Ptr{Cvoid}, (Ptr{ggml_context},), ctx)
end

"""
    ggml_get_mem_size(ctx)


### Prototype
```c
size_t ggml_get_mem_size (const struct ggml_context * ctx);
```
"""
function ggml_get_mem_size(ctx)
    ccall((:ggml_get_mem_size, libllama), Csize_t, (Ptr{ggml_context},), ctx)
end

"""
    ggml_get_max_tensor_size(ctx)


### Prototype
```c
size_t ggml_get_max_tensor_size(const struct ggml_context * ctx);
```
"""
function ggml_get_max_tensor_size(ctx)
    ccall((:ggml_get_max_tensor_size, libllama), Csize_t, (Ptr{ggml_context},), ctx)
end

"""
    ggml_new_tensor(ctx, type, n_dims, ne)


### Prototype
```c
struct ggml_tensor * ggml_new_tensor( struct ggml_context * ctx, enum ggml_type type, int n_dims, const int64_t *ne);
```
"""
function ggml_new_tensor(ctx, type, n_dims, ne)
    ccall((:ggml_new_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Cint, Ptr{Int64}), ctx, type, n_dims, ne)
end

"""
    ggml_new_tensor_1d(ctx, type, ne0)


### Prototype
```c
struct ggml_tensor * ggml_new_tensor_1d( struct ggml_context * ctx, enum ggml_type type, int64_t ne0);
```
"""
function ggml_new_tensor_1d(ctx, type, ne0)
    ccall((:ggml_new_tensor_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64), ctx, type, ne0)
end

"""
    ggml_new_tensor_2d(ctx, type, ne0, ne1)


### Prototype
```c
struct ggml_tensor * ggml_new_tensor_2d( struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1);
```
"""
function ggml_new_tensor_2d(ctx, type, ne0, ne1)
    ccall((:ggml_new_tensor_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64, Int64), ctx, type, ne0, ne1)
end

"""
    ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2)


### Prototype
```c
struct ggml_tensor * ggml_new_tensor_3d( struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2);
```
"""
function ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2)
    ccall((:ggml_new_tensor_3d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64, Int64, Int64), ctx, type, ne0, ne1, ne2)
end

"""
    ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3)


### Prototype
```c
struct ggml_tensor * ggml_new_tensor_4d( struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
```
"""
function ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3)
    ccall((:ggml_new_tensor_4d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64, Int64, Int64, Int64), ctx, type, ne0, ne1, ne2, ne3)
end

"""
    ggml_new_i32(ctx, value)


### Prototype
```c
struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
```
"""
function ggml_new_i32(ctx, value)
    ccall((:ggml_new_i32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Int32), ctx, value)
end

"""
    ggml_new_f32(ctx, value)


### Prototype
```c
struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);
```
"""
function ggml_new_f32(ctx, value)
    ccall((:ggml_new_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Cfloat), ctx, value)
end

"""
    ggml_dup_tensor(ctx, src)


### Prototype
```c
struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
```
"""
function ggml_dup_tensor(ctx, src)
    ccall((:ggml_dup_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, src)
end

"""
    ggml_view_tensor(ctx, src)


### Prototype
```c
struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);
```
"""
function ggml_view_tensor(ctx, src)
    ccall((:ggml_view_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, src)
end

"""
    ggml_get_first_tensor(ctx)


### Prototype
```c
struct ggml_tensor * ggml_get_first_tensor(const struct ggml_context * ctx);
```
"""
function ggml_get_first_tensor(ctx)
    ccall((:ggml_get_first_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context},), ctx)
end

"""
    ggml_get_next_tensor(ctx, tensor)


### Prototype
```c
struct ggml_tensor * ggml_get_next_tensor (const struct ggml_context * ctx, struct ggml_tensor * tensor);
```
"""
function ggml_get_next_tensor(ctx, tensor)
    ccall((:ggml_get_next_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, tensor)
end

"""
    ggml_get_tensor(ctx, name)


### Prototype
```c
struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name);
```
"""
function ggml_get_tensor(ctx, name)
    ccall((:ggml_get_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{Cchar}), ctx, name)
end

"""
    ggml_set_zero(tensor)


### Prototype
```c
struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
```
"""
function ggml_set_zero(tensor)
    ccall((:ggml_set_zero, libllama), Ptr{ggml_tensor}, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_set_i32(tensor, value)


### Prototype
```c
struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
```
"""
function ggml_set_i32(tensor, value)
    ccall((:ggml_set_i32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_tensor}, Int32), tensor, value)
end

"""
    ggml_set_f32(tensor, value)


### Prototype
```c
struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);
```
"""
function ggml_set_f32(tensor, value)
    ccall((:ggml_set_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_tensor}, Cfloat), tensor, value)
end

"""
    ggml_unravel_index(tensor, i, i0, i1, i2, i3)


### Prototype
```c
void ggml_unravel_index(const struct ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);
```
"""
function ggml_unravel_index(tensor, i, i0, i1, i2, i3)
    ccall((:ggml_unravel_index, libllama), Cvoid, (Ptr{ggml_tensor}, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}), tensor, i, i0, i1, i2, i3)
end

"""
    ggml_get_i32_1d(tensor, i)


### Prototype
```c
int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
```
"""
function ggml_get_i32_1d(tensor, i)
    ccall((:ggml_get_i32_1d, libllama), Int32, (Ptr{ggml_tensor}, Cint), tensor, i)
end

"""
    ggml_set_i32_1d(tensor, i, value)


### Prototype
```c
void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);
```
"""
function ggml_set_i32_1d(tensor, i, value)
    ccall((:ggml_set_i32_1d, libllama), Cvoid, (Ptr{ggml_tensor}, Cint, Int32), tensor, i, value)
end

"""
    ggml_get_i32_nd(tensor, i0, i1, i2, i3)


### Prototype
```c
int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
```
"""
function ggml_get_i32_nd(tensor, i0, i1, i2, i3)
    ccall((:ggml_get_i32_nd, libllama), Int32, (Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), tensor, i0, i1, i2, i3)
end

"""
    ggml_set_i32_nd(tensor, i0, i1, i2, i3, value)


### Prototype
```c
void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);
```
"""
function ggml_set_i32_nd(tensor, i0, i1, i2, i3, value)
    ccall((:ggml_set_i32_nd, libllama), Cvoid, (Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Int32), tensor, i0, i1, i2, i3, value)
end

"""
    ggml_get_f32_1d(tensor, i)


### Prototype
```c
float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
```
"""
function ggml_get_f32_1d(tensor, i)
    ccall((:ggml_get_f32_1d, libllama), Cfloat, (Ptr{ggml_tensor}, Cint), tensor, i)
end

"""
    ggml_set_f32_1d(tensor, i, value)


### Prototype
```c
void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);
```
"""
function ggml_set_f32_1d(tensor, i, value)
    ccall((:ggml_set_f32_1d, libllama), Cvoid, (Ptr{ggml_tensor}, Cint, Cfloat), tensor, i, value)
end

"""
    ggml_get_f32_nd(tensor, i0, i1, i2, i3)


### Prototype
```c
float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
```
"""
function ggml_get_f32_nd(tensor, i0, i1, i2, i3)
    ccall((:ggml_get_f32_nd, libllama), Cfloat, (Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), tensor, i0, i1, i2, i3)
end

"""
    ggml_set_f32_nd(tensor, i0, i1, i2, i3, value)


### Prototype
```c
void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);
```
"""
function ggml_set_f32_nd(tensor, i0, i1, i2, i3, value)
    ccall((:ggml_set_f32_nd, libllama), Cvoid, (Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cfloat), tensor, i0, i1, i2, i3, value)
end

"""
    ggml_get_data(tensor)


### Prototype
```c
void * ggml_get_data (const struct ggml_tensor * tensor);
```
"""
function ggml_get_data(tensor)
    ccall((:ggml_get_data, libllama), Ptr{Cvoid}, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_get_data_f32(tensor)


### Prototype
```c
float * ggml_get_data_f32(const struct ggml_tensor * tensor);
```
"""
function ggml_get_data_f32(tensor)
    ccall((:ggml_get_data_f32, libllama), Ptr{Cfloat}, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_get_unary_op(tensor)


### Prototype
```c
enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * tensor);
```
"""
function ggml_get_unary_op(tensor)
    ccall((:ggml_get_unary_op, libllama), ggml_unary_op, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_get_name(tensor)


### Prototype
```c
const char * ggml_get_name (const struct ggml_tensor * tensor);
```
"""
function ggml_get_name(tensor)
    ccall((:ggml_get_name, libllama), Ptr{Cchar}, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_set_name(tensor, name)


### Prototype
```c
struct ggml_tensor * ggml_set_name ( struct ggml_tensor * tensor, const char * name);
```
"""
function ggml_set_name(tensor, name)
    ccall((:ggml_set_name, libllama), Ptr{ggml_tensor}, (Ptr{ggml_tensor}, Ptr{Cchar}), tensor, name)
end

"""
    ggml_dup(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_dup( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_dup(ctx, a)
    ccall((:ggml_dup, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_dup_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_dup_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_dup_inplace(ctx, a)
    ccall((:ggml_dup_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_add(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_add( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_add(ctx, a, b)
    ccall((:ggml_add, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_add_inplace(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_add_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_add_inplace(ctx, a, b)
    ccall((:ggml_add_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_add_cast(ctx, a, b, type)


### Prototype
```c
struct ggml_tensor * ggml_add_cast( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, enum ggml_type type);
```
"""
function ggml_add_cast(ctx, a, b, type)
    ccall((:ggml_add_cast, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_type), ctx, a, b, type)
end

"""
    ggml_add1(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_add1( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_add1(ctx, a, b)
    ccall((:ggml_add1, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_add1_inplace(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_add1_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_add1_inplace(ctx, a, b)
    ccall((:ggml_add1_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_acc(ctx, a, b, nb1, nb2, nb3, offset)


### Prototype
```c
struct ggml_tensor * ggml_acc( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t nb2, size_t nb3, size_t offset);
```
"""
function ggml_acc(ctx, a, b, nb1, nb2, nb3, offset)
    ccall((:ggml_acc, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, b, nb1, nb2, nb3, offset)
end

"""
    ggml_acc_inplace(ctx, a, b, nb1, nb2, nb3, offset)


### Prototype
```c
struct ggml_tensor * ggml_acc_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t nb2, size_t nb3, size_t offset);
```
"""
function ggml_acc_inplace(ctx, a, b, nb1, nb2, nb3, offset)
    ccall((:ggml_acc_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, b, nb1, nb2, nb3, offset)
end

"""
    ggml_sub(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_sub( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_sub(ctx, a, b)
    ccall((:ggml_sub, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_sub_inplace(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_sub_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_sub_inplace(ctx, a, b)
    ccall((:ggml_sub_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_mul(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_mul( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_mul(ctx, a, b)
    ccall((:ggml_mul, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_mul_inplace(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_mul_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_mul_inplace(ctx, a, b)
    ccall((:ggml_mul_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_div(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_div( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_div(ctx, a, b)
    ccall((:ggml_div, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_div_inplace(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_div_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_div_inplace(ctx, a, b)
    ccall((:ggml_div_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_sqr(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_sqr( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_sqr(ctx, a)
    ccall((:ggml_sqr, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_sqr_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_sqr_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_sqr_inplace(ctx, a)
    ccall((:ggml_sqr_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_sqrt(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_sqrt( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_sqrt(ctx, a)
    ccall((:ggml_sqrt, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_sqrt_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_sqrt_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_sqrt_inplace(ctx, a)
    ccall((:ggml_sqrt_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_log(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_log( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_log(ctx, a)
    ccall((:ggml_log, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_log_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_log_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_log_inplace(ctx, a)
    ccall((:ggml_log_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_sum(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_sum( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_sum(ctx, a)
    ccall((:ggml_sum, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_sum_rows(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_sum_rows( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_sum_rows(ctx, a)
    ccall((:ggml_sum_rows, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_mean(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_mean( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_mean(ctx, a)
    ccall((:ggml_mean, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_argmax(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_argmax( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_argmax(ctx, a)
    ccall((:ggml_argmax, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_repeat(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_repeat( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_repeat(ctx, a, b)
    ccall((:ggml_repeat, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_repeat_back(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_repeat_back( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_repeat_back(ctx, a, b)
    ccall((:ggml_repeat_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_concat(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_concat( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_concat(ctx, a, b)
    ccall((:ggml_concat, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_abs(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_abs( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_abs(ctx, a)
    ccall((:ggml_abs, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_abs_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_abs_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_abs_inplace(ctx, a)
    ccall((:ggml_abs_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_sgn(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_sgn( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_sgn(ctx, a)
    ccall((:ggml_sgn, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_sgn_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_sgn_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_sgn_inplace(ctx, a)
    ccall((:ggml_sgn_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_neg(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_neg( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_neg(ctx, a)
    ccall((:ggml_neg, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_neg_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_neg_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_neg_inplace(ctx, a)
    ccall((:ggml_neg_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_step(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_step( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_step(ctx, a)
    ccall((:ggml_step, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_step_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_step_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_step_inplace(ctx, a)
    ccall((:ggml_step_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_tanh(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_tanh( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_tanh(ctx, a)
    ccall((:ggml_tanh, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_tanh_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_tanh_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_tanh_inplace(ctx, a)
    ccall((:ggml_tanh_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_elu(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_elu( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_elu(ctx, a)
    ccall((:ggml_elu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_elu_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_elu_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_elu_inplace(ctx, a)
    ccall((:ggml_elu_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_relu(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_relu( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_relu(ctx, a)
    ccall((:ggml_relu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_leaky_relu(ctx, a, negative_slope, inplace)


### Prototype
```c
struct ggml_tensor * ggml_leaky_relu( struct ggml_context * ctx, struct ggml_tensor * a, float negative_slope, bool inplace);
```
"""
function ggml_leaky_relu(ctx, a, negative_slope, inplace)
    ccall((:ggml_leaky_relu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat, Bool), ctx, a, negative_slope, inplace)
end

"""
    ggml_relu_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_relu_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_relu_inplace(ctx, a)
    ccall((:ggml_relu_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_gelu(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_gelu( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_gelu(ctx, a)
    ccall((:ggml_gelu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_gelu_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_gelu_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_gelu_inplace(ctx, a)
    ccall((:ggml_gelu_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_gelu_quick(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_gelu_quick( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_gelu_quick(ctx, a)
    ccall((:ggml_gelu_quick, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_gelu_quick_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_gelu_quick_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_gelu_quick_inplace(ctx, a)
    ccall((:ggml_gelu_quick_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_silu(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_silu( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_silu(ctx, a)
    ccall((:ggml_silu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_silu_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_silu_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_silu_inplace(ctx, a)
    ccall((:ggml_silu_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_silu_back(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_silu_back( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_silu_back(ctx, a, b)
    ccall((:ggml_silu_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_norm(ctx, a, eps)


### Prototype
```c
struct ggml_tensor * ggml_norm( struct ggml_context * ctx, struct ggml_tensor * a, float eps);
```
"""
function ggml_norm(ctx, a, eps)
    ccall((:ggml_norm, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

"""
    ggml_norm_inplace(ctx, a, eps)


### Prototype
```c
struct ggml_tensor * ggml_norm_inplace( struct ggml_context * ctx, struct ggml_tensor * a, float eps);
```
"""
function ggml_norm_inplace(ctx, a, eps)
    ccall((:ggml_norm_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

"""
    ggml_rms_norm(ctx, a, eps)


### Prototype
```c
struct ggml_tensor * ggml_rms_norm( struct ggml_context * ctx, struct ggml_tensor * a, float eps);
```
"""
function ggml_rms_norm(ctx, a, eps)
    ccall((:ggml_rms_norm, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

"""
    ggml_rms_norm_inplace(ctx, a, eps)


### Prototype
```c
struct ggml_tensor * ggml_rms_norm_inplace( struct ggml_context * ctx, struct ggml_tensor * a, float eps);
```
"""
function ggml_rms_norm_inplace(ctx, a, eps)
    ccall((:ggml_rms_norm_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

"""
    ggml_group_norm(ctx, a, n_groups)


### Prototype
```c
struct ggml_tensor * ggml_group_norm( struct ggml_context * ctx, struct ggml_tensor * a, int n_groups);
```
"""
function ggml_group_norm(ctx, a, n_groups)
    ccall((:ggml_group_norm, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_groups)
end

"""
    ggml_group_norm_inplace(ctx, a, n_groups)


### Prototype
```c
struct ggml_tensor * ggml_group_norm_inplace( struct ggml_context * ctx, struct ggml_tensor * a, int n_groups);
```
"""
function ggml_group_norm_inplace(ctx, a, n_groups)
    ccall((:ggml_group_norm_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_groups)
end

"""
    ggml_rms_norm_back(ctx, a, b, eps)


### Prototype
```c
struct ggml_tensor * ggml_rms_norm_back( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, float eps);
```
"""
function ggml_rms_norm_back(ctx, a, b, eps)
    ccall((:ggml_rms_norm_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat), ctx, a, b, eps)
end

"""
    ggml_mul_mat(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_mul_mat( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_mul_mat(ctx, a, b)
    ccall((:ggml_mul_mat, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_mul_mat_set_prec(a, prec)


### Prototype
```c
void ggml_mul_mat_set_prec( struct ggml_tensor * a, enum ggml_prec prec);
```
"""
function ggml_mul_mat_set_prec(a, prec)
    ccall((:ggml_mul_mat_set_prec, libllama), Cvoid, (Ptr{ggml_tensor}, ggml_prec), a, prec)
end

"""
    ggml_mul_mat_id(ctx, as, n_as, ids, id, b)


### Prototype
```c
struct ggml_tensor * ggml_mul_mat_id( struct ggml_context * ctx, struct ggml_tensor * const as[], int n_as, struct ggml_tensor * ids, int id, struct ggml_tensor * b);
```
"""
function ggml_mul_mat_id(ctx, as, n_as, ids, id, b)
    ccall((:ggml_mul_mat_id, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{Ptr{ggml_tensor}}, Cint, Ptr{ggml_tensor}, Cint, Ptr{ggml_tensor}), ctx, as, n_as, ids, id, b)
end

"""
    ggml_out_prod(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_out_prod( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_out_prod(ctx, a, b)
    ccall((:ggml_out_prod, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_scale(ctx, a, s)


### Prototype
```c
struct ggml_tensor * ggml_scale( struct ggml_context * ctx, struct ggml_tensor * a, float s);
```
"""
function ggml_scale(ctx, a, s)
    ccall((:ggml_scale, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, s)
end

"""
    ggml_scale_inplace(ctx, a, s)


### Prototype
```c
struct ggml_tensor * ggml_scale_inplace( struct ggml_context * ctx, struct ggml_tensor * a, float s);
```
"""
function ggml_scale_inplace(ctx, a, s)
    ccall((:ggml_scale_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, s)
end

"""
    ggml_set(ctx, a, b, nb1, nb2, nb3, offset)


### Prototype
```c
struct ggml_tensor * ggml_set( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t nb2, size_t nb3, size_t offset);
```
"""
function ggml_set(ctx, a, b, nb1, nb2, nb3, offset)
    ccall((:ggml_set, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, b, nb1, nb2, nb3, offset)
end

"""
    ggml_set_inplace(ctx, a, b, nb1, nb2, nb3, offset)


### Prototype
```c
struct ggml_tensor * ggml_set_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t nb2, size_t nb3, size_t offset);
```
"""
function ggml_set_inplace(ctx, a, b, nb1, nb2, nb3, offset)
    ccall((:ggml_set_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, b, nb1, nb2, nb3, offset)
end

"""
    ggml_set_1d(ctx, a, b, offset)


### Prototype
```c
struct ggml_tensor * ggml_set_1d( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t offset);
```
"""
function ggml_set_1d(ctx, a, b, offset)
    ccall((:ggml_set_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t), ctx, a, b, offset)
end

"""
    ggml_set_1d_inplace(ctx, a, b, offset)


### Prototype
```c
struct ggml_tensor * ggml_set_1d_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t offset);
```
"""
function ggml_set_1d_inplace(ctx, a, b, offset)
    ccall((:ggml_set_1d_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t), ctx, a, b, offset)
end

"""
    ggml_set_2d(ctx, a, b, nb1, offset)


### Prototype
```c
struct ggml_tensor * ggml_set_2d( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t offset);
```
"""
function ggml_set_2d(ctx, a, b, nb1, offset)
    ccall((:ggml_set_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t), ctx, a, b, nb1, offset)
end

"""
    ggml_set_2d_inplace(ctx, a, b, nb1, offset)


### Prototype
```c
struct ggml_tensor * ggml_set_2d_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t offset);
```
"""
function ggml_set_2d_inplace(ctx, a, b, nb1, offset)
    ccall((:ggml_set_2d_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t), ctx, a, b, nb1, offset)
end

"""
    ggml_cpy(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_cpy( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_cpy(ctx, a, b)
    ccall((:ggml_cpy, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_cpy_inplace(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_cpy_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_cpy_inplace(ctx, a, b)
    ccall((:ggml_cpy_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_cont(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_cont( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_cont(ctx, a)
    ccall((:ggml_cont, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_cont_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_cont_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_cont_inplace(ctx, a)
    ccall((:ggml_cont_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_cont_1d(ctx, a, ne0)


### Prototype
```c
struct ggml_tensor * ggml_cont_1d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0);
```
"""
function ggml_cont_1d(ctx, a, ne0)
    ccall((:ggml_cont_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64), ctx, a, ne0)
end

"""
    ggml_cont_2d(ctx, a, ne0, ne1)


### Prototype
```c
struct ggml_tensor * ggml_cont_2d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1);
```
"""
function ggml_cont_2d(ctx, a, ne0, ne1)
    ccall((:ggml_cont_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64), ctx, a, ne0, ne1)
end

"""
    ggml_cont_3d(ctx, a, ne0, ne1, ne2)


### Prototype
```c
struct ggml_tensor * ggml_cont_3d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2);
```
"""
function ggml_cont_3d(ctx, a, ne0, ne1, ne2)
    ccall((:ggml_cont_3d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2)
end

"""
    ggml_cont_4d(ctx, a, ne0, ne1, ne2, ne3)


### Prototype
```c
struct ggml_tensor * ggml_cont_4d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
```
"""
function ggml_cont_4d(ctx, a, ne0, ne1, ne2, ne3)
    ccall((:ggml_cont_4d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2, ne3)
end

"""
    ggml_reshape(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_reshape( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_reshape(ctx, a, b)
    ccall((:ggml_reshape, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_reshape_1d(ctx, a, ne0)


### Prototype
```c
struct ggml_tensor * ggml_reshape_1d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0);
```
"""
function ggml_reshape_1d(ctx, a, ne0)
    ccall((:ggml_reshape_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64), ctx, a, ne0)
end

"""
    ggml_reshape_2d(ctx, a, ne0, ne1)


### Prototype
```c
struct ggml_tensor * ggml_reshape_2d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1);
```
"""
function ggml_reshape_2d(ctx, a, ne0, ne1)
    ccall((:ggml_reshape_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64), ctx, a, ne0, ne1)
end

"""
    ggml_reshape_3d(ctx, a, ne0, ne1, ne2)


### Prototype
```c
struct ggml_tensor * ggml_reshape_3d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2);
```
"""
function ggml_reshape_3d(ctx, a, ne0, ne1, ne2)
    ccall((:ggml_reshape_3d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2)
end

"""
    ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3)


### Prototype
```c
struct ggml_tensor * ggml_reshape_4d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
```
"""
function ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3)
    ccall((:ggml_reshape_4d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2, ne3)
end

"""
    ggml_view_1d(ctx, a, ne0, offset)


### Prototype
```c
struct ggml_tensor * ggml_view_1d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, size_t offset);
```
"""
function ggml_view_1d(ctx, a, ne0, offset)
    ccall((:ggml_view_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Csize_t), ctx, a, ne0, offset)
end

"""
    ggml_view_2d(ctx, a, ne0, ne1, nb1, offset)


### Prototype
```c
struct ggml_tensor * ggml_view_2d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
```
"""
function ggml_view_2d(ctx, a, ne0, ne1, nb1, offset)
    ccall((:ggml_view_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Csize_t, Csize_t), ctx, a, ne0, ne1, nb1, offset)
end

"""
    ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset)


### Prototype
```c
struct ggml_tensor * ggml_view_3d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset);
```
"""
function ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset)
    ccall((:ggml_view_3d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Csize_t, Csize_t, Csize_t), ctx, a, ne0, ne1, ne2, nb1, nb2, offset)
end

"""
    ggml_view_4d(ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset)


### Prototype
```c
struct ggml_tensor * ggml_view_4d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset);
```
"""
function ggml_view_4d(ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset)
    ccall((:ggml_view_4d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset)
end

"""
    ggml_permute(ctx, a, axis0, axis1, axis2, axis3)


### Prototype
```c
struct ggml_tensor * ggml_permute( struct ggml_context * ctx, struct ggml_tensor * a, int axis0, int axis1, int axis2, int axis3);
```
"""
function ggml_permute(ctx, a, axis0, axis1, axis2, axis3)
    ccall((:ggml_permute, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, axis0, axis1, axis2, axis3)
end

"""
    ggml_transpose(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_transpose( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_transpose(ctx, a)
    ccall((:ggml_transpose, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_get_rows(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_get_rows( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_get_rows(ctx, a, b)
    ccall((:ggml_get_rows, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_get_rows_back(ctx, a, b, c)


### Prototype
```c
struct ggml_tensor * ggml_get_rows_back( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c);
```
"""
function ggml_get_rows_back(ctx, a, b, c)
    ccall((:ggml_get_rows_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b, c)
end

"""
    ggml_diag(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_diag( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_diag(ctx, a)
    ccall((:ggml_diag, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_diag_mask_inf(ctx, a, n_past)


### Prototype
```c
struct ggml_tensor * ggml_diag_mask_inf( struct ggml_context * ctx, struct ggml_tensor * a, int n_past);
```
"""
function ggml_diag_mask_inf(ctx, a, n_past)
    ccall((:ggml_diag_mask_inf, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_past)
end

"""
    ggml_diag_mask_inf_inplace(ctx, a, n_past)


### Prototype
```c
struct ggml_tensor * ggml_diag_mask_inf_inplace( struct ggml_context * ctx, struct ggml_tensor * a, int n_past);
```
"""
function ggml_diag_mask_inf_inplace(ctx, a, n_past)
    ccall((:ggml_diag_mask_inf_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_past)
end

"""
    ggml_diag_mask_zero(ctx, a, n_past)


### Prototype
```c
struct ggml_tensor * ggml_diag_mask_zero( struct ggml_context * ctx, struct ggml_tensor * a, int n_past);
```
"""
function ggml_diag_mask_zero(ctx, a, n_past)
    ccall((:ggml_diag_mask_zero, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_past)
end

"""
    ggml_diag_mask_zero_inplace(ctx, a, n_past)


### Prototype
```c
struct ggml_tensor * ggml_diag_mask_zero_inplace( struct ggml_context * ctx, struct ggml_tensor * a, int n_past);
```
"""
function ggml_diag_mask_zero_inplace(ctx, a, n_past)
    ccall((:ggml_diag_mask_zero_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_past)
end

"""
    ggml_soft_max(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_soft_max( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_soft_max(ctx, a)
    ccall((:ggml_soft_max, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_soft_max_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_soft_max_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_soft_max_inplace(ctx, a)
    ccall((:ggml_soft_max_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_soft_max_ext(ctx, a, mask, scale)


### Prototype
```c
struct ggml_tensor * ggml_soft_max_ext( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * mask, float scale);
```
"""
function ggml_soft_max_ext(ctx, a, mask, scale)
    ccall((:ggml_soft_max_ext, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat), ctx, a, mask, scale)
end

"""
    ggml_soft_max_back(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_soft_max_back( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_soft_max_back(ctx, a, b)
    ccall((:ggml_soft_max_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_soft_max_back_inplace(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_soft_max_back_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_soft_max_back_inplace(ctx, a, b)
    ccall((:ggml_soft_max_back_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_rope(ctx, a, b, n_dims, mode, n_ctx)


### Prototype
```c
struct ggml_tensor * ggml_rope( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int n_dims, int mode, int n_ctx);
```
"""
function ggml_rope(ctx, a, b, n_dims, mode, n_ctx)
    ccall((:ggml_rope, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, b, n_dims, mode, n_ctx)
end

"""
    ggml_rope_inplace(ctx, a, b, n_dims, mode, n_ctx)


### Prototype
```c
struct ggml_tensor * ggml_rope_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int n_dims, int mode, int n_ctx);
```
"""
function ggml_rope_inplace(ctx, a, b, n_dims, mode, n_ctx)
    ccall((:ggml_rope_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, b, n_dims, mode, n_ctx)
end

"""
    ggml_rope_custom(ctx, a, b, n_dims, mode, n_ctx, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)


### Prototype
```c
struct ggml_tensor * ggml_rope_custom( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int n_dims, int mode, int n_ctx, int n_orig_ctx, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
```
"""
function ggml_rope_custom(ctx, a, b, n_dims, mode, n_ctx, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_custom, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, n_dims, mode, n_ctx, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

"""
    ggml_rope_custom_inplace(ctx, a, b, n_dims, mode, n_ctx, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)


### Prototype
```c
struct ggml_tensor * ggml_rope_custom_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int n_dims, int mode, int n_ctx, int n_orig_ctx, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
```
"""
function ggml_rope_custom_inplace(ctx, a, b, n_dims, mode, n_ctx, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_custom_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, n_dims, mode, n_ctx, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

"""
    ggml_rope_yarn_corr_dims(n_dims, n_orig_ctx, freq_base, beta_fast, beta_slow, dims)


### Prototype
```c
void ggml_rope_yarn_corr_dims( int n_dims, int n_orig_ctx, float freq_base, float beta_fast, float beta_slow, float dims[2]);
```
"""
function ggml_rope_yarn_corr_dims(n_dims, n_orig_ctx, freq_base, beta_fast, beta_slow, dims)
    ccall((:ggml_rope_yarn_corr_dims, libllama), Cvoid, (Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}), n_dims, n_orig_ctx, freq_base, beta_fast, beta_slow, dims)
end

"""
    ggml_rope_xpos_inplace(ctx, a, b, n_dims, base, down)


### Prototype
```c
struct ggml_tensor * ggml_rope_xpos_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int n_dims, float base, bool down);
```
"""
function ggml_rope_xpos_inplace(ctx, a, b, n_dims, base, down)
    ccall((:ggml_rope_xpos_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cfloat, Bool), ctx, a, b, n_dims, base, down)
end

"""
    ggml_rope_back(ctx, a, b, n_dims, mode, n_ctx, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, xpos_base, xpos_down)


### Prototype
```c
struct ggml_tensor * ggml_rope_back( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int n_dims, int mode, int n_ctx, int n_orig_ctx, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float xpos_base, bool xpos_down);
```
"""
function ggml_rope_back(ctx, a, b, n_dims, mode, n_ctx, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, xpos_base, xpos_down)
    ccall((:ggml_rope_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Bool), ctx, a, b, n_dims, mode, n_ctx, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, xpos_base, xpos_down)
end

"""
    ggml_alibi(ctx, a, n_past, n_head, bias_max)


### Prototype
```c
struct ggml_tensor * ggml_alibi( struct ggml_context * ctx, struct ggml_tensor * a, int n_past, int n_head, float bias_max);
```
"""
function ggml_alibi(ctx, a, n_past, n_head, bias_max)
    ccall((:ggml_alibi, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cfloat), ctx, a, n_past, n_head, bias_max)
end

"""
    ggml_clamp(ctx, a, min, max)


### Prototype
```c
struct ggml_tensor * ggml_clamp( struct ggml_context * ctx, struct ggml_tensor * a, float min, float max);
```
"""
function ggml_clamp(ctx, a, min, max)
    ccall((:ggml_clamp, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, min, max)
end

"""
    ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D)


### Prototype
```c
struct ggml_tensor * ggml_im2col( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int p0, int p1, int d0, int d1, bool is_2D);
```
"""
function ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D)
    ccall((:ggml_im2col, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint, Bool), ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D)
end

"""
    ggml_conv_1d(ctx, a, b, s0, p0, d0)


### Prototype
```c
struct ggml_tensor * ggml_conv_1d( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int p0, int d0);
```
"""
function ggml_conv_1d(ctx, a, b, s0, p0, d0)
    ccall((:ggml_conv_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, b, s0, p0, d0)
end

"""
    ggml_conv_1d_ph(ctx, a, b, s, d)


### Prototype
```c
struct ggml_tensor* ggml_conv_1d_ph( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s, int d);
```
"""
function ggml_conv_1d_ph(ctx, a, b, s, d)
    ccall((:ggml_conv_1d_ph, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, b, s, d)
end

"""
    ggml_conv_transpose_1d(ctx, a, b, s0, p0, d0)


### Prototype
```c
struct ggml_tensor * ggml_conv_transpose_1d( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int p0, int d0);
```
"""
function ggml_conv_transpose_1d(ctx, a, b, s0, p0, d0)
    ccall((:ggml_conv_transpose_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, b, s0, p0, d0)
end

"""
    ggml_conv_2d(ctx, a, b, s0, s1, p0, p1, d0, d1)


### Prototype
```c
struct ggml_tensor * ggml_conv_2d( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int p0, int p1, int d0, int d1);
```
"""
function ggml_conv_2d(ctx, a, b, s0, s1, p0, p1, d0, d1)
    ccall((:ggml_conv_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, s0, s1, p0, p1, d0, d1)
end

"""
    ggml_conv_2d_sk_p0(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_conv_2d_sk_p0( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_conv_2d_sk_p0(ctx, a, b)
    ccall((:ggml_conv_2d_sk_p0, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_conv_2d_s1_ph(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_conv_2d_s1_ph( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_conv_2d_s1_ph(ctx, a, b)
    ccall((:ggml_conv_2d_s1_ph, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_conv_transpose_2d_p0(ctx, a, b, stride)


### Prototype
```c
struct ggml_tensor * ggml_conv_transpose_2d_p0( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int stride);
```
"""
function ggml_conv_transpose_2d_p0(ctx, a, b, stride)
    ccall((:ggml_conv_transpose_2d_p0, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint), ctx, a, b, stride)
end

@cenum ggml_op_pool::UInt32 begin
    GGML_OP_POOL_MAX = 0
    GGML_OP_POOL_AVG = 1
    GGML_OP_POOL_COUNT = 2
end

"""
    ggml_pool_1d(ctx, a, op, k0, s0, p0)


### Prototype
```c
struct ggml_tensor * ggml_pool_1d( struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_op_pool op, int k0, int s0, int p0);
```
"""
function ggml_pool_1d(ctx, a, op, k0, s0, p0)
    ccall((:ggml_pool_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_op_pool, Cint, Cint, Cint), ctx, a, op, k0, s0, p0)
end

"""
    ggml_pool_2d(ctx, a, op, k0, k1, s0, s1, p0, p1)


### Prototype
```c
struct ggml_tensor * ggml_pool_2d( struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1);
```
"""
function ggml_pool_2d(ctx, a, op, k0, k1, s0, s1, p0, p1)
    ccall((:ggml_pool_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_op_pool, Cint, Cint, Cint, Cint, Cfloat, Cfloat), ctx, a, op, k0, k1, s0, s1, p0, p1)
end

"""
    ggml_upscale(ctx, a, scale_factor)


### Prototype
```c
struct ggml_tensor * ggml_upscale( struct ggml_context * ctx, struct ggml_tensor * a, int scale_factor);
```
"""
function ggml_upscale(ctx, a, scale_factor)
    ccall((:ggml_upscale, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, scale_factor)
end

"""
    ggml_pad(ctx, a, p0, p1, p2, p3)


### Prototype
```c
struct ggml_tensor * ggml_pad( struct ggml_context * ctx, struct ggml_tensor * a, int p0, int p1, int p2, int p3);
```
"""
function ggml_pad(ctx, a, p0, p1, p2, p3)
    ccall((:ggml_pad, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, p0, p1, p2, p3)
end

@cenum ggml_sort_order::UInt32 begin
    GGML_SORT_ASC = 0
    GGML_SORT_DESC = 1
end

"""
    ggml_argsort(ctx, a, order)


### Prototype
```c
struct ggml_tensor * ggml_argsort( struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_sort_order order);
```
"""
function ggml_argsort(ctx, a, order)
    ccall((:ggml_argsort, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_sort_order), ctx, a, order)
end

"""
    ggml_top_k(ctx, a, k)


### Prototype
```c
struct ggml_tensor * ggml_top_k( struct ggml_context * ctx, struct ggml_tensor * a, int k);
```
"""
function ggml_top_k(ctx, a, k)
    ccall((:ggml_top_k, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, k)
end

"""
    ggml_flash_attn(ctx, q, k, v, masked)


### Prototype
```c
struct ggml_tensor * ggml_flash_attn( struct ggml_context * ctx, struct ggml_tensor * q, struct ggml_tensor * k, struct ggml_tensor * v, bool masked);
```
"""
function ggml_flash_attn(ctx, q, k, v, masked)
    ccall((:ggml_flash_attn, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Bool), ctx, q, k, v, masked)
end

"""
    ggml_flash_attn_back(ctx, q, k, v, d, masked)


### Prototype
```c
struct ggml_tensor * ggml_flash_attn_back( struct ggml_context * ctx, struct ggml_tensor * q, struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor * d, bool masked);
```
"""
function ggml_flash_attn_back(ctx, q, k, v, d, masked)
    ccall((:ggml_flash_attn_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Bool), ctx, q, k, v, d, masked)
end

"""
    ggml_flash_ff(ctx, a, b0, b1, c0, c1)


### Prototype
```c
struct ggml_tensor * ggml_flash_ff( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b0, struct ggml_tensor * b1, struct ggml_tensor * c0, struct ggml_tensor * c1);
```
"""
function ggml_flash_ff(ctx, a, b0, b1, c0, c1)
    ccall((:ggml_flash_ff, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b0, b1, c0, c1)
end

"""
    ggml_win_part(ctx, a, w)


### Prototype
```c
struct ggml_tensor * ggml_win_part( struct ggml_context * ctx, struct ggml_tensor * a, int w);
```
"""
function ggml_win_part(ctx, a, w)
    ccall((:ggml_win_part, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, w)
end

"""
    ggml_win_unpart(ctx, a, w0, h0, w)


### Prototype
```c
struct ggml_tensor * ggml_win_unpart( struct ggml_context * ctx, struct ggml_tensor * a, int w0, int h0, int w);
```
"""
function ggml_win_unpart(ctx, a, w0, h0, w)
    ccall((:ggml_win_unpart, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, w0, h0, w)
end

"""
    ggml_unary(ctx, a, op)


### Prototype
```c
struct ggml_tensor * ggml_unary( struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_unary_op op);
```
"""
function ggml_unary(ctx, a, op)
    ccall((:ggml_unary, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_unary_op), ctx, a, op)
end

"""
    ggml_unary_inplace(ctx, a, op)


### Prototype
```c
struct ggml_tensor * ggml_unary_inplace( struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_unary_op op);
```
"""
function ggml_unary_inplace(ctx, a, op)
    ccall((:ggml_unary_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_unary_op), ctx, a, op)
end

"""
    ggml_get_rel_pos(ctx, a, qh, kh)


### Prototype
```c
struct ggml_tensor * ggml_get_rel_pos( struct ggml_context * ctx, struct ggml_tensor * a, int qh, int kh);
```
"""
function ggml_get_rel_pos(ctx, a, qh, kh)
    ccall((:ggml_get_rel_pos, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, qh, kh)
end

"""
    ggml_add_rel_pos(ctx, a, pw, ph)


### Prototype
```c
struct ggml_tensor * ggml_add_rel_pos( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * pw, struct ggml_tensor * ph);
```
"""
function ggml_add_rel_pos(ctx, a, pw, ph)
    ccall((:ggml_add_rel_pos, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, pw, ph)
end

"""
    ggml_add_rel_pos_inplace(ctx, a, pw, ph)


### Prototype
```c
struct ggml_tensor * ggml_add_rel_pos_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * pw, struct ggml_tensor * ph);
```
"""
function ggml_add_rel_pos_inplace(ctx, a, pw, ph)
    ccall((:ggml_add_rel_pos_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, pw, ph)
end

# typedef void ( * ggml_unary_op_f32_t ) ( const int , float * , const float * )
const ggml_unary_op_f32_t = Ptr{Cvoid}

# typedef void ( * ggml_binary_op_f32_t ) ( const int , float * , const float * , const float * )
const ggml_binary_op_f32_t = Ptr{Cvoid}

# typedef void ( * ggml_custom1_op_f32_t ) ( struct ggml_tensor * , const struct ggml_tensor * )
const ggml_custom1_op_f32_t = Ptr{Cvoid}

# typedef void ( * ggml_custom2_op_f32_t ) ( struct ggml_tensor * , const struct ggml_tensor * , const struct ggml_tensor * )
const ggml_custom2_op_f32_t = Ptr{Cvoid}

# typedef void ( * ggml_custom3_op_f32_t ) ( struct ggml_tensor * , const struct ggml_tensor * , const struct ggml_tensor * , const struct ggml_tensor * )
const ggml_custom3_op_f32_t = Ptr{Cvoid}

"""
    ggml_map_unary_f32(ctx, a, fun)


### Prototype
```c
GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_unary_f32( struct ggml_context * ctx, struct ggml_tensor * a, ggml_unary_op_f32_t fun), "use ggml_map_custom1 instead");
```
"""
function ggml_map_unary_f32(ctx, a, fun)
    ccall((:ggml_map_unary_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_unary_op_f32_t), ctx, a, fun)
end

"""
    ggml_map_unary_inplace_f32(ctx, a, fun)


### Prototype
```c
GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_unary_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, ggml_unary_op_f32_t fun), "use ggml_map_custom1_inplace instead");
```
"""
function ggml_map_unary_inplace_f32(ctx, a, fun)
    ccall((:ggml_map_unary_inplace_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_unary_op_f32_t), ctx, a, fun)
end

"""
    ggml_map_binary_f32(ctx, a, b, fun)


### Prototype
```c
GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_binary_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_binary_op_f32_t fun), "use ggml_map_custom2 instead");
```
"""
function ggml_map_binary_f32(ctx, a, b, fun)
    ccall((:ggml_map_binary_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_binary_op_f32_t), ctx, a, b, fun)
end

"""
    ggml_map_binary_inplace_f32(ctx, a, b, fun)


### Prototype
```c
GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_binary_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_binary_op_f32_t fun), "use ggml_map_custom2_inplace instead");
```
"""
function ggml_map_binary_inplace_f32(ctx, a, b, fun)
    ccall((:ggml_map_binary_inplace_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_binary_op_f32_t), ctx, a, b, fun)
end

"""
    ggml_map_custom1_f32(ctx, a, fun)


### Prototype
```c
GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom1_f32( struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_f32_t fun), "use ggml_map_custom1 instead");
```
"""
function ggml_map_custom1_f32(ctx, a, fun)
    ccall((:ggml_map_custom1_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_custom1_op_f32_t), ctx, a, fun)
end

"""
    ggml_map_custom1_inplace_f32(ctx, a, fun)


### Prototype
```c
GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom1_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_f32_t fun), "use ggml_map_custom1_inplace instead");
```
"""
function ggml_map_custom1_inplace_f32(ctx, a, fun)
    ccall((:ggml_map_custom1_inplace_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_custom1_op_f32_t), ctx, a, fun)
end

"""
    ggml_map_custom2_f32(ctx, a, b, fun)


### Prototype
```c
GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom2_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_f32_t fun), "use ggml_map_custom2 instead");
```
"""
function ggml_map_custom2_f32(ctx, a, b, fun)
    ccall((:ggml_map_custom2_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom2_op_f32_t), ctx, a, b, fun)
end

"""
    ggml_map_custom2_inplace_f32(ctx, a, b, fun)


### Prototype
```c
GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom2_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_f32_t fun), "use ggml_map_custom2_inplace instead");
```
"""
function ggml_map_custom2_inplace_f32(ctx, a, b, fun)
    ccall((:ggml_map_custom2_inplace_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom2_op_f32_t), ctx, a, b, fun)
end

"""
    ggml_map_custom3_f32(ctx, a, b, c, fun)


### Prototype
```c
GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom3_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_f32_t fun), "use ggml_map_custom3 instead");
```
"""
function ggml_map_custom3_f32(ctx, a, b, c, fun)
    ccall((:ggml_map_custom3_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom3_op_f32_t), ctx, a, b, c, fun)
end

"""
    ggml_map_custom3_inplace_f32(ctx, a, b, c, fun)


### Prototype
```c
GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom3_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_f32_t fun), "use ggml_map_custom3_inplace instead");
```
"""
function ggml_map_custom3_inplace_f32(ctx, a, b, c, fun)
    ccall((:ggml_map_custom3_inplace_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom3_op_f32_t), ctx, a, b, c, fun)
end

# typedef void ( * ggml_custom1_op_t ) ( struct ggml_tensor * dst , const struct ggml_tensor * a , int ith , int nth , void * userdata )
const ggml_custom1_op_t = Ptr{Cvoid}

# typedef void ( * ggml_custom2_op_t ) ( struct ggml_tensor * dst , const struct ggml_tensor * a , const struct ggml_tensor * b , int ith , int nth , void * userdata )
const ggml_custom2_op_t = Ptr{Cvoid}

# typedef void ( * ggml_custom3_op_t ) ( struct ggml_tensor * dst , const struct ggml_tensor * a , const struct ggml_tensor * b , const struct ggml_tensor * c , int ith , int nth , void * userdata )
const ggml_custom3_op_t = Ptr{Cvoid}

"""
    ggml_map_custom1(ctx, a, fun, n_tasks, userdata)


### Prototype
```c
struct ggml_tensor * ggml_map_custom1( struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_t fun, int n_tasks, void * userdata);
```
"""
function ggml_map_custom1(ctx, a, fun, n_tasks, userdata)
    ccall((:ggml_map_custom1, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_custom1_op_t, Cint, Ptr{Cvoid}), ctx, a, fun, n_tasks, userdata)
end

"""
    ggml_map_custom1_inplace(ctx, a, fun, n_tasks, userdata)


### Prototype
```c
struct ggml_tensor * ggml_map_custom1_inplace( struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_t fun, int n_tasks, void * userdata);
```
"""
function ggml_map_custom1_inplace(ctx, a, fun, n_tasks, userdata)
    ccall((:ggml_map_custom1_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_custom1_op_t, Cint, Ptr{Cvoid}), ctx, a, fun, n_tasks, userdata)
end

"""
    ggml_map_custom2(ctx, a, b, fun, n_tasks, userdata)


### Prototype
```c
struct ggml_tensor * ggml_map_custom2( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_t fun, int n_tasks, void * userdata);
```
"""
function ggml_map_custom2(ctx, a, b, fun, n_tasks, userdata)
    ccall((:ggml_map_custom2, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom2_op_t, Cint, Ptr{Cvoid}), ctx, a, b, fun, n_tasks, userdata)
end

"""
    ggml_map_custom2_inplace(ctx, a, b, fun, n_tasks, userdata)


### Prototype
```c
struct ggml_tensor * ggml_map_custom2_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_t fun, int n_tasks, void * userdata);
```
"""
function ggml_map_custom2_inplace(ctx, a, b, fun, n_tasks, userdata)
    ccall((:ggml_map_custom2_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom2_op_t, Cint, Ptr{Cvoid}), ctx, a, b, fun, n_tasks, userdata)
end

"""
    ggml_map_custom3(ctx, a, b, c, fun, n_tasks, userdata)


### Prototype
```c
struct ggml_tensor * ggml_map_custom3( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_t fun, int n_tasks, void * userdata);
```
"""
function ggml_map_custom3(ctx, a, b, c, fun, n_tasks, userdata)
    ccall((:ggml_map_custom3, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom3_op_t, Cint, Ptr{Cvoid}), ctx, a, b, c, fun, n_tasks, userdata)
end

"""
    ggml_map_custom3_inplace(ctx, a, b, c, fun, n_tasks, userdata)


### Prototype
```c
struct ggml_tensor * ggml_map_custom3_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_t fun, int n_tasks, void * userdata);
```
"""
function ggml_map_custom3_inplace(ctx, a, b, c, fun, n_tasks, userdata)
    ccall((:ggml_map_custom3_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom3_op_t, Cint, Ptr{Cvoid}), ctx, a, b, c, fun, n_tasks, userdata)
end

"""
    ggml_cross_entropy_loss(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_cross_entropy_loss( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_cross_entropy_loss(ctx, a, b)
    ccall((:ggml_cross_entropy_loss, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_cross_entropy_loss_back(ctx, a, b, c)


### Prototype
```c
struct ggml_tensor * ggml_cross_entropy_loss_back( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c);
```
"""
function ggml_cross_entropy_loss_back(ctx, a, b, c)
    ccall((:ggml_cross_entropy_loss_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b, c)
end

"""
    ggml_set_param(ctx, tensor)


### Prototype
```c
void ggml_set_param( struct ggml_context * ctx, struct ggml_tensor * tensor);
```
"""
function ggml_set_param(ctx, tensor)
    ccall((:ggml_set_param, libllama), Cvoid, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, tensor)
end

"""
    ggml_build_forward_expand(cgraph, tensor)


### Prototype
```c
void ggml_build_forward_expand (struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
```
"""
function ggml_build_forward_expand(cgraph, tensor)
    ccall((:ggml_build_forward_expand, libllama), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_tensor}), cgraph, tensor)
end

"""
    ggml_build_backward_expand(ctx, gf, gb, keep)


### Prototype
```c
void ggml_build_backward_expand(struct ggml_context * ctx, struct ggml_cgraph * gf, struct ggml_cgraph * gb, bool keep);
```
"""
function ggml_build_backward_expand(ctx, gf, gb, keep)
    ccall((:ggml_build_backward_expand, libllama), Cvoid, (Ptr{ggml_context}, Ptr{ggml_cgraph}, Ptr{ggml_cgraph}, Bool), ctx, gf, gb, keep)
end

"""
    ggml_new_graph(ctx)


### Prototype
```c
struct ggml_cgraph * ggml_new_graph (struct ggml_context * ctx);
```
"""
function ggml_new_graph(ctx)
    ccall((:ggml_new_graph, libllama), Ptr{ggml_cgraph}, (Ptr{ggml_context},), ctx)
end

"""
    ggml_new_graph_custom(ctx, size, grads)


### Prototype
```c
struct ggml_cgraph * ggml_new_graph_custom (struct ggml_context * ctx, size_t size, bool grads);
```
"""
function ggml_new_graph_custom(ctx, size, grads)
    ccall((:ggml_new_graph_custom, libllama), Ptr{ggml_cgraph}, (Ptr{ggml_context}, Csize_t, Bool), ctx, size, grads)
end

"""
    ggml_graph_dup(ctx, cgraph)


### Prototype
```c
struct ggml_cgraph * ggml_graph_dup (struct ggml_context * ctx, struct ggml_cgraph * cgraph);
```
"""
function ggml_graph_dup(ctx, cgraph)
    ccall((:ggml_graph_dup, libllama), Ptr{ggml_cgraph}, (Ptr{ggml_context}, Ptr{ggml_cgraph}), ctx, cgraph)
end

"""
    ggml_graph_view(cgraph, i0, i1)


### Prototype
```c
struct ggml_cgraph ggml_graph_view (struct ggml_cgraph * cgraph, int i0, int i1);
```
"""
function ggml_graph_view(cgraph, i0, i1)
    ccall((:ggml_graph_view, libllama), ggml_cgraph, (Ptr{ggml_cgraph}, Cint, Cint), cgraph, i0, i1)
end

"""
    ggml_graph_cpy(src, dst)


### Prototype
```c
void ggml_graph_cpy (struct ggml_cgraph * src, struct ggml_cgraph * dst);
```
"""
function ggml_graph_cpy(src, dst)
    ccall((:ggml_graph_cpy, libllama), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_cgraph}), src, dst)
end

"""
    ggml_graph_reset(cgraph)


### Prototype
```c
void ggml_graph_reset (struct ggml_cgraph * cgraph);
```
"""
function ggml_graph_reset(cgraph)
    ccall((:ggml_graph_reset, libllama), Cvoid, (Ptr{ggml_cgraph},), cgraph)
end

"""
    ggml_graph_clear(cgraph)


### Prototype
```c
void ggml_graph_clear (struct ggml_cgraph * cgraph);
```
"""
function ggml_graph_clear(cgraph)
    ccall((:ggml_graph_clear, libllama), Cvoid, (Ptr{ggml_cgraph},), cgraph)
end

"""
    ggml_graph_overhead()


### Prototype
```c
size_t ggml_graph_overhead(void);
```
"""
function ggml_graph_overhead()
    ccall((:ggml_graph_overhead, libllama), Csize_t, ())
end

"""
    ggml_graph_overhead_custom(size, grads)


### Prototype
```c
size_t ggml_graph_overhead_custom(size_t size, bool grads);
```
"""
function ggml_graph_overhead_custom(size, grads)
    ccall((:ggml_graph_overhead_custom, libllama), Csize_t, (Csize_t, Bool), size, grads)
end

"""
    ggml_graph_plan(cgraph, n_threads)


### Prototype
```c
struct ggml_cplan ggml_graph_plan (struct ggml_cgraph * cgraph, int n_threads );
```
"""
function ggml_graph_plan(cgraph, n_threads)
    ccall((:ggml_graph_plan, libllama), ggml_cplan, (Ptr{ggml_cgraph}, Cint), cgraph, n_threads)
end

"""
    ggml_graph_compute(cgraph, cplan)


### Prototype
```c
int ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
```
"""
function ggml_graph_compute(cgraph, cplan)
    ccall((:ggml_graph_compute, libllama), Cint, (Ptr{ggml_cgraph}, Ptr{ggml_cplan}), cgraph, cplan)
end

"""
    ggml_graph_compute_with_ctx(ctx, cgraph, n_threads)


### Prototype
```c
void ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
```
"""
function ggml_graph_compute_with_ctx(ctx, cgraph, n_threads)
    ccall((:ggml_graph_compute_with_ctx, libllama), Cvoid, (Ptr{ggml_context}, Ptr{ggml_cgraph}, Cint), ctx, cgraph, n_threads)
end

"""
    ggml_graph_get_tensor(cgraph, name)


### Prototype
```c
struct ggml_tensor * ggml_graph_get_tensor(struct ggml_cgraph * cgraph, const char * name);
```
"""
function ggml_graph_get_tensor(cgraph, name)
    ccall((:ggml_graph_get_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_cgraph}, Ptr{Cchar}), cgraph, name)
end

"""
    ggml_graph_export(cgraph, fname)


### Prototype
```c
void ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname);
```
"""
function ggml_graph_export(cgraph, fname)
    ccall((:ggml_graph_export, libllama), Cvoid, (Ptr{ggml_cgraph}, Ptr{Cchar}), cgraph, fname)
end

"""
    ggml_graph_import(fname, ctx_data, ctx_eval)


### Prototype
```c
struct ggml_cgraph * ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval);
```
"""
function ggml_graph_import(fname, ctx_data, ctx_eval)
    ccall((:ggml_graph_import, libllama), Ptr{ggml_cgraph}, (Ptr{Cchar}, Ptr{Ptr{ggml_context}}, Ptr{Ptr{ggml_context}}), fname, ctx_data, ctx_eval)
end

"""
    ggml_graph_print(cgraph)


### Prototype
```c
void ggml_graph_print(const struct ggml_cgraph * cgraph);
```
"""
function ggml_graph_print(cgraph)
    ccall((:ggml_graph_print, libllama), Cvoid, (Ptr{ggml_cgraph},), cgraph)
end

"""
    ggml_graph_dump_dot(gb, gf, filename)


### Prototype
```c
void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);
```
"""
function ggml_graph_dump_dot(gb, gf, filename)
    ccall((:ggml_graph_dump_dot, libllama), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_cgraph}, Ptr{Cchar}), gb, gf, filename)
end

"""
    ggml_build_backward_gradient_checkpointing(ctx, gf, gb, gb_tmp, checkpoints, n_checkpoints)


### Prototype
```c
void ggml_build_backward_gradient_checkpointing( struct ggml_context * ctx, struct ggml_cgraph * gf, struct ggml_cgraph * gb, struct ggml_cgraph * gb_tmp, struct ggml_tensor * * checkpoints, int n_checkpoints);
```
"""
function ggml_build_backward_gradient_checkpointing(ctx, gf, gb, gb_tmp, checkpoints, n_checkpoints)
    ccall((:ggml_build_backward_gradient_checkpointing, libllama), Cvoid, (Ptr{ggml_context}, Ptr{ggml_cgraph}, Ptr{ggml_cgraph}, Ptr{ggml_cgraph}, Ptr{Ptr{ggml_tensor}}, Cint), ctx, gf, gb, gb_tmp, checkpoints, n_checkpoints)
end

@cenum ggml_opt_type::UInt32 begin
    GGML_OPT_ADAM = 0
    GGML_OPT_LBFGS = 1
end

@cenum ggml_linesearch::UInt32 begin
    GGML_LINESEARCH_DEFAULT = 1
    GGML_LINESEARCH_BACKTRACKING_ARMIJO = 0
    GGML_LINESEARCH_BACKTRACKING_WOLFE = 1
    GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2
end

@cenum ggml_opt_result::Int32 begin
    GGML_OPT_OK = 0
    GGML_OPT_DID_NOT_CONVERGE = 1
    GGML_OPT_NO_CONTEXT = 2
    GGML_OPT_INVALID_WOLFE = 3
    GGML_OPT_FAIL = 4
    GGML_OPT_CANCEL = 5
    GGML_LINESEARCH_FAIL = -128
    GGML_LINESEARCH_MINIMUM_STEP = -127
    GGML_LINESEARCH_MAXIMUM_STEP = -126
    GGML_LINESEARCH_MAXIMUM_ITERATIONS = -125
    GGML_LINESEARCH_INVALID_PARAMETERS = -124
end

# typedef void ( * ggml_opt_callback ) ( void * data , int accum_step , float * sched , bool * cancel )
const ggml_opt_callback = Ptr{Cvoid}

struct __JL_Ctag_10
    n_iter::Cint
    sched::Cfloat
    decay::Cfloat
    decay_min_ndim::Cint
    alpha::Cfloat
    beta1::Cfloat
    beta2::Cfloat
    eps::Cfloat
    eps_f::Cfloat
    eps_g::Cfloat
    gclip::Cfloat
end
function Base.getproperty(x::Ptr{__JL_Ctag_10}, f::Symbol)
    f === :n_iter && return Ptr{Cint}(x + 0)
    f === :sched && return Ptr{Cfloat}(x + 4)
    f === :decay && return Ptr{Cfloat}(x + 8)
    f === :decay_min_ndim && return Ptr{Cint}(x + 12)
    f === :alpha && return Ptr{Cfloat}(x + 16)
    f === :beta1 && return Ptr{Cfloat}(x + 20)
    f === :beta2 && return Ptr{Cfloat}(x + 24)
    f === :eps && return Ptr{Cfloat}(x + 28)
    f === :eps_f && return Ptr{Cfloat}(x + 32)
    f === :eps_g && return Ptr{Cfloat}(x + 36)
    f === :gclip && return Ptr{Cfloat}(x + 40)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_10, f::Symbol)
    r = Ref{__JL_Ctag_10}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_10}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_10}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct __JL_Ctag_11
    m::Cint
    n_iter::Cint
    max_linesearch::Cint
    eps::Cfloat
    ftol::Cfloat
    wolfe::Cfloat
    min_step::Cfloat
    max_step::Cfloat
    linesearch::ggml_linesearch
end
function Base.getproperty(x::Ptr{__JL_Ctag_11}, f::Symbol)
    f === :m && return Ptr{Cint}(x + 0)
    f === :n_iter && return Ptr{Cint}(x + 4)
    f === :max_linesearch && return Ptr{Cint}(x + 8)
    f === :eps && return Ptr{Cfloat}(x + 12)
    f === :ftol && return Ptr{Cfloat}(x + 16)
    f === :wolfe && return Ptr{Cfloat}(x + 20)
    f === :min_step && return Ptr{Cfloat}(x + 24)
    f === :max_step && return Ptr{Cfloat}(x + 28)
    f === :linesearch && return Ptr{ggml_linesearch}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_11, f::Symbol)
    r = Ref{__JL_Ctag_11}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_11}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_11}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct ggml_opt_params
    data::NTuple{120, UInt8}
end

function Base.getproperty(x::Ptr{ggml_opt_params}, f::Symbol)
    f === :type && return Ptr{ggml_opt_type}(x + 0)
    f === :graph_size && return Ptr{Csize_t}(x + 8)
    f === :n_threads && return Ptr{Cint}(x + 16)
    f === :past && return Ptr{Cint}(x + 20)
    f === :delta && return Ptr{Cfloat}(x + 24)
    f === :max_no_improvement && return Ptr{Cint}(x + 28)
    f === :print_forward_graph && return Ptr{Bool}(x + 32)
    f === :print_backward_graph && return Ptr{Bool}(x + 33)
    f === :n_gradient_accumulation && return Ptr{Cint}(x + 36)
    f === :adam && return Ptr{__JL_Ctag_10}(x + 40)
    f === :lbfgs && return Ptr{__JL_Ctag_11}(x + 84)
    return getfield(x, f)
end

function Base.getproperty(x::ggml_opt_params, f::Symbol)
    r = Ref{ggml_opt_params}(x)
    ptr = Base.unsafe_convert(Ptr{ggml_opt_params}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{ggml_opt_params}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct __JL_Ctag_8
    g::Ptr{ggml_tensor}
    m::Ptr{ggml_tensor}
    v::Ptr{ggml_tensor}
    pf::Ptr{ggml_tensor}
    fx_best::Cfloat
    fx_prev::Cfloat
    n_no_improvement::Cint
end
function Base.getproperty(x::Ptr{__JL_Ctag_8}, f::Symbol)
    f === :g && return Ptr{Ptr{ggml_tensor}}(x + 0)
    f === :m && return Ptr{Ptr{ggml_tensor}}(x + 8)
    f === :v && return Ptr{Ptr{ggml_tensor}}(x + 16)
    f === :pf && return Ptr{Ptr{ggml_tensor}}(x + 24)
    f === :fx_best && return Ptr{Cfloat}(x + 32)
    f === :fx_prev && return Ptr{Cfloat}(x + 36)
    f === :n_no_improvement && return Ptr{Cint}(x + 40)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_8, f::Symbol)
    r = Ref{__JL_Ctag_8}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_8}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_8}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct __JL_Ctag_9
    x::Ptr{ggml_tensor}
    xp::Ptr{ggml_tensor}
    g::Ptr{ggml_tensor}
    gp::Ptr{ggml_tensor}
    d::Ptr{ggml_tensor}
    pf::Ptr{ggml_tensor}
    lmal::Ptr{ggml_tensor}
    lmys::Ptr{ggml_tensor}
    lms::Ptr{ggml_tensor}
    lmy::Ptr{ggml_tensor}
    fx_best::Cfloat
    step::Cfloat
    j::Cint
    k::Cint
    _end::Cint
    n_no_improvement::Cint
end
function Base.getproperty(x::Ptr{__JL_Ctag_9}, f::Symbol)
    f === :x && return Ptr{Ptr{ggml_tensor}}(x + 0)
    f === :xp && return Ptr{Ptr{ggml_tensor}}(x + 8)
    f === :g && return Ptr{Ptr{ggml_tensor}}(x + 16)
    f === :gp && return Ptr{Ptr{ggml_tensor}}(x + 24)
    f === :d && return Ptr{Ptr{ggml_tensor}}(x + 32)
    f === :pf && return Ptr{Ptr{ggml_tensor}}(x + 40)
    f === :lmal && return Ptr{Ptr{ggml_tensor}}(x + 48)
    f === :lmys && return Ptr{Ptr{ggml_tensor}}(x + 56)
    f === :lms && return Ptr{Ptr{ggml_tensor}}(x + 64)
    f === :lmy && return Ptr{Ptr{ggml_tensor}}(x + 72)
    f === :fx_best && return Ptr{Cfloat}(x + 80)
    f === :step && return Ptr{Cfloat}(x + 84)
    f === :j && return Ptr{Cint}(x + 88)
    f === :k && return Ptr{Cint}(x + 92)
    f === :_end && return Ptr{Cint}(x + 96)
    f === :n_no_improvement && return Ptr{Cint}(x + 100)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_9, f::Symbol)
    r = Ref{__JL_Ctag_9}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_9}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_9}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct ggml_opt_context
    data::NTuple{312, UInt8}
end

function Base.getproperty(x::Ptr{ggml_opt_context}, f::Symbol)
    f === :ctx && return Ptr{Ptr{ggml_context}}(x + 0)
    f === :params && return Ptr{ggml_opt_params}(x + 8)
    f === :iter && return Ptr{Cint}(x + 128)
    f === :nx && return Ptr{Int64}(x + 136)
    f === :just_initialized && return Ptr{Bool}(x + 144)
    f === :loss_before && return Ptr{Cfloat}(x + 148)
    f === :loss_after && return Ptr{Cfloat}(x + 152)
    f === :adam && return Ptr{__JL_Ctag_8}(x + 160)
    f === :lbfgs && return Ptr{__JL_Ctag_9}(x + 208)
    return getfield(x, f)
end

function Base.getproperty(x::ggml_opt_context, f::Symbol)
    r = Ref{ggml_opt_context}(x)
    ptr = Base.unsafe_convert(Ptr{ggml_opt_context}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{ggml_opt_context}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

"""
    ggml_opt_default_params(type)


### Prototype
```c
struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type);
```
"""
function ggml_opt_default_params(type)
    ccall((:ggml_opt_default_params, libllama), ggml_opt_params, (ggml_opt_type,), type)
end

"""
    ggml_opt(ctx, params, f)


### Prototype
```c
enum ggml_opt_result ggml_opt( struct ggml_context * ctx, struct ggml_opt_params params, struct ggml_tensor * f);
```
"""
function ggml_opt(ctx, params, f)
    ccall((:ggml_opt, libllama), ggml_opt_result, (Ptr{ggml_context}, ggml_opt_params, Ptr{ggml_tensor}), ctx, params, f)
end

"""
    ggml_opt_init(ctx, opt, params, nx)


### Prototype
```c
void ggml_opt_init( struct ggml_context * ctx, struct ggml_opt_context * opt, struct ggml_opt_params params, int64_t nx);
```
"""
function ggml_opt_init(ctx, opt, params, nx)
    ccall((:ggml_opt_init, libllama), Cvoid, (Ptr{ggml_context}, Ptr{ggml_opt_context}, ggml_opt_params, Int64), ctx, opt, params, nx)
end

"""
    ggml_opt_resume(ctx, opt, f)


### Prototype
```c
enum ggml_opt_result ggml_opt_resume( struct ggml_context * ctx, struct ggml_opt_context * opt, struct ggml_tensor * f);
```
"""
function ggml_opt_resume(ctx, opt, f)
    ccall((:ggml_opt_resume, libllama), ggml_opt_result, (Ptr{ggml_context}, Ptr{ggml_opt_context}, Ptr{ggml_tensor}), ctx, opt, f)
end

"""
    ggml_opt_resume_g(ctx, opt, f, gf, gb, callback, callback_data)


### Prototype
```c
enum ggml_opt_result ggml_opt_resume_g( struct ggml_context * ctx, struct ggml_opt_context * opt, struct ggml_tensor * f, struct ggml_cgraph * gf, struct ggml_cgraph * gb, ggml_opt_callback callback, void * callback_data);
```
"""
function ggml_opt_resume_g(ctx, opt, f, gf, gb, callback, callback_data)
    ccall((:ggml_opt_resume_g, libllama), ggml_opt_result, (Ptr{ggml_context}, Ptr{ggml_opt_context}, Ptr{ggml_tensor}, Ptr{ggml_cgraph}, Ptr{ggml_cgraph}, ggml_opt_callback, Ptr{Cvoid}), ctx, opt, f, gf, gb, callback, callback_data)
end

"""
    ggml_quantize_q4_0(src, dst, n, k, hist)


### Prototype
```c
size_t ggml_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
```
"""
function ggml_quantize_q4_0(src, dst, n, k, hist)
    ccall((:ggml_quantize_q4_0, libllama), Csize_t, (Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), src, dst, n, k, hist)
end

"""
    ggml_quantize_q4_1(src, dst, n, k, hist)


### Prototype
```c
size_t ggml_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);
```
"""
function ggml_quantize_q4_1(src, dst, n, k, hist)
    ccall((:ggml_quantize_q4_1, libllama), Csize_t, (Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), src, dst, n, k, hist)
end

"""
    ggml_quantize_q5_0(src, dst, n, k, hist)


### Prototype
```c
size_t ggml_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist);
```
"""
function ggml_quantize_q5_0(src, dst, n, k, hist)
    ccall((:ggml_quantize_q5_0, libllama), Csize_t, (Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), src, dst, n, k, hist)
end

"""
    ggml_quantize_q5_1(src, dst, n, k, hist)


### Prototype
```c
size_t ggml_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist);
```
"""
function ggml_quantize_q5_1(src, dst, n, k, hist)
    ccall((:ggml_quantize_q5_1, libllama), Csize_t, (Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), src, dst, n, k, hist)
end

"""
    ggml_quantize_q8_0(src, dst, n, k, hist)


### Prototype
```c
size_t ggml_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist);
```
"""
function ggml_quantize_q8_0(src, dst, n, k, hist)
    ccall((:ggml_quantize_q8_0, libllama), Csize_t, (Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), src, dst, n, k, hist)
end

"""
    ggml_quantize_q2_K(src, dst, n, k, hist)


### Prototype
```c
size_t ggml_quantize_q2_K(const float * src, void * dst, int n, int k, int64_t * hist);
```
"""
function ggml_quantize_q2_K(src, dst, n, k, hist)
    ccall((:ggml_quantize_q2_K, libllama), Csize_t, (Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), src, dst, n, k, hist)
end

"""
    ggml_quantize_q3_K(src, dst, n, k, hist)


### Prototype
```c
size_t ggml_quantize_q3_K(const float * src, void * dst, int n, int k, int64_t * hist);
```
"""
function ggml_quantize_q3_K(src, dst, n, k, hist)
    ccall((:ggml_quantize_q3_K, libllama), Csize_t, (Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), src, dst, n, k, hist)
end

"""
    ggml_quantize_q4_K(src, dst, n, k, hist)


### Prototype
```c
size_t ggml_quantize_q4_K(const float * src, void * dst, int n, int k, int64_t * hist);
```
"""
function ggml_quantize_q4_K(src, dst, n, k, hist)
    ccall((:ggml_quantize_q4_K, libllama), Csize_t, (Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), src, dst, n, k, hist)
end

"""
    ggml_quantize_q5_K(src, dst, n, k, hist)


### Prototype
```c
size_t ggml_quantize_q5_K(const float * src, void * dst, int n, int k, int64_t * hist);
```
"""
function ggml_quantize_q5_K(src, dst, n, k, hist)
    ccall((:ggml_quantize_q5_K, libllama), Csize_t, (Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), src, dst, n, k, hist)
end

"""
    ggml_quantize_q6_K(src, dst, n, k, hist)


### Prototype
```c
size_t ggml_quantize_q6_K(const float * src, void * dst, int n, int k, int64_t * hist);
```
"""
function ggml_quantize_q6_K(src, dst, n, k, hist)
    ccall((:ggml_quantize_q6_K, libllama), Csize_t, (Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), src, dst, n, k, hist)
end

"""
    ggml_quantize_iq2_xxs(src, dst, n, k, hist)


### Prototype
```c
size_t ggml_quantize_iq2_xxs(const float * src, void * dst, int n, int k, int64_t * hist);
```
"""
function ggml_quantize_iq2_xxs(src, dst, n, k, hist)
    ccall((:ggml_quantize_iq2_xxs, libllama), Csize_t, (Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), src, dst, n, k, hist)
end

"""
    ggml_quantize_chunk(type, src, dst, start, n, hist)


### Prototype
```c
size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst, int start, int n, int64_t * hist);
```
"""
function ggml_quantize_chunk(type, src, dst, start, n, hist)
    ccall((:ggml_quantize_chunk, libllama), Csize_t, (ggml_type, Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), type, src, dst, start, n, hist)
end

@cenum gguf_type::UInt32 begin
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12
    GGUF_TYPE_COUNT = 13
end

mutable struct gguf_context end

struct gguf_init_params
    no_alloc::Bool
    ctx::Ptr{Ptr{ggml_context}}
end

"""
    gguf_init_empty()


### Prototype
```c
struct gguf_context * gguf_init_empty(void);
```
"""
function gguf_init_empty()
    ccall((:gguf_init_empty, libllama), Ptr{gguf_context}, ())
end

"""
    gguf_init_from_file(fname, params)


### Prototype
```c
struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);
```
"""
function gguf_init_from_file(fname, params)
    ccall((:gguf_init_from_file, libllama), Ptr{gguf_context}, (Ptr{Cchar}, gguf_init_params), fname, params)
end

"""
    gguf_free(ctx)


### Prototype
```c
void gguf_free(struct gguf_context * ctx);
```
"""
function gguf_free(ctx)
    ccall((:gguf_free, libllama), Cvoid, (Ptr{gguf_context},), ctx)
end

"""
    gguf_type_name(type)


### Prototype
```c
const char * gguf_type_name(enum gguf_type type);
```
"""
function gguf_type_name(type)
    ccall((:gguf_type_name, libllama), Ptr{Cchar}, (gguf_type,), type)
end

"""
    gguf_get_version(ctx)


### Prototype
```c
int gguf_get_version (const struct gguf_context * ctx);
```
"""
function gguf_get_version(ctx)
    ccall((:gguf_get_version, libllama), Cint, (Ptr{gguf_context},), ctx)
end

"""
    gguf_get_alignment(ctx)


### Prototype
```c
size_t gguf_get_alignment (const struct gguf_context * ctx);
```
"""
function gguf_get_alignment(ctx)
    ccall((:gguf_get_alignment, libllama), Csize_t, (Ptr{gguf_context},), ctx)
end

"""
    gguf_get_data_offset(ctx)


### Prototype
```c
size_t gguf_get_data_offset(const struct gguf_context * ctx);
```
"""
function gguf_get_data_offset(ctx)
    ccall((:gguf_get_data_offset, libllama), Csize_t, (Ptr{gguf_context},), ctx)
end

"""
    gguf_get_data(ctx)


### Prototype
```c
void * gguf_get_data (const struct gguf_context * ctx);
```
"""
function gguf_get_data(ctx)
    ccall((:gguf_get_data, libllama), Ptr{Cvoid}, (Ptr{gguf_context},), ctx)
end

"""
    gguf_get_n_kv(ctx)


### Prototype
```c
int gguf_get_n_kv(const struct gguf_context * ctx);
```
"""
function gguf_get_n_kv(ctx)
    ccall((:gguf_get_n_kv, libllama), Cint, (Ptr{gguf_context},), ctx)
end

"""
    gguf_find_key(ctx, key)


### Prototype
```c
int gguf_find_key(const struct gguf_context * ctx, const char * key);
```
"""
function gguf_find_key(ctx, key)
    ccall((:gguf_find_key, libllama), Cint, (Ptr{gguf_context}, Ptr{Cchar}), ctx, key)
end

"""
    gguf_get_key(ctx, key_id)


### Prototype
```c
const char * gguf_get_key (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_key(ctx, key_id)
    ccall((:gguf_get_key, libllama), Ptr{Cchar}, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_kv_type(ctx, key_id)


### Prototype
```c
enum gguf_type gguf_get_kv_type (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_kv_type(ctx, key_id)
    ccall((:gguf_get_kv_type, libllama), gguf_type, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_arr_type(ctx, key_id)


### Prototype
```c
enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_arr_type(ctx, key_id)
    ccall((:gguf_get_arr_type, libllama), gguf_type, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_u8(ctx, key_id)


### Prototype
```c
uint8_t gguf_get_val_u8 (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_u8(ctx, key_id)
    ccall((:gguf_get_val_u8, libllama), UInt8, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_i8(ctx, key_id)


### Prototype
```c
int8_t gguf_get_val_i8 (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_i8(ctx, key_id)
    ccall((:gguf_get_val_i8, libllama), Int8, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_u16(ctx, key_id)


### Prototype
```c
uint16_t gguf_get_val_u16 (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_u16(ctx, key_id)
    ccall((:gguf_get_val_u16, libllama), UInt16, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_i16(ctx, key_id)


### Prototype
```c
int16_t gguf_get_val_i16 (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_i16(ctx, key_id)
    ccall((:gguf_get_val_i16, libllama), Int16, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_u32(ctx, key_id)


### Prototype
```c
uint32_t gguf_get_val_u32 (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_u32(ctx, key_id)
    ccall((:gguf_get_val_u32, libllama), UInt32, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_i32(ctx, key_id)


### Prototype
```c
int32_t gguf_get_val_i32 (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_i32(ctx, key_id)
    ccall((:gguf_get_val_i32, libllama), Int32, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_f32(ctx, key_id)


### Prototype
```c
float gguf_get_val_f32 (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_f32(ctx, key_id)
    ccall((:gguf_get_val_f32, libllama), Cfloat, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_u64(ctx, key_id)


### Prototype
```c
uint64_t gguf_get_val_u64 (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_u64(ctx, key_id)
    ccall((:gguf_get_val_u64, libllama), UInt64, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_i64(ctx, key_id)


### Prototype
```c
int64_t gguf_get_val_i64 (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_i64(ctx, key_id)
    ccall((:gguf_get_val_i64, libllama), Int64, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_f64(ctx, key_id)


### Prototype
```c
double gguf_get_val_f64 (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_f64(ctx, key_id)
    ccall((:gguf_get_val_f64, libllama), Cdouble, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_bool(ctx, key_id)


### Prototype
```c
bool gguf_get_val_bool(const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_bool(ctx, key_id)
    ccall((:gguf_get_val_bool, libllama), Bool, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_str(ctx, key_id)


### Prototype
```c
const char * gguf_get_val_str (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_str(ctx, key_id)
    ccall((:gguf_get_val_str, libllama), Ptr{Cchar}, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_val_data(ctx, key_id)


### Prototype
```c
const void * gguf_get_val_data(const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_val_data(ctx, key_id)
    ccall((:gguf_get_val_data, libllama), Ptr{Cvoid}, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_arr_n(ctx, key_id)


### Prototype
```c
int gguf_get_arr_n (const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_arr_n(ctx, key_id)
    ccall((:gguf_get_arr_n, libllama), Cint, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_arr_data(ctx, key_id)


### Prototype
```c
const void * gguf_get_arr_data(const struct gguf_context * ctx, int key_id);
```
"""
function gguf_get_arr_data(ctx, key_id)
    ccall((:gguf_get_arr_data, libllama), Ptr{Cvoid}, (Ptr{gguf_context}, Cint), ctx, key_id)
end

"""
    gguf_get_arr_str(ctx, key_id, i)


### Prototype
```c
const char * gguf_get_arr_str (const struct gguf_context * ctx, int key_id, int i);
```
"""
function gguf_get_arr_str(ctx, key_id, i)
    ccall((:gguf_get_arr_str, libllama), Ptr{Cchar}, (Ptr{gguf_context}, Cint, Cint), ctx, key_id, i)
end

"""
    gguf_get_n_tensors(ctx)


### Prototype
```c
int gguf_get_n_tensors (const struct gguf_context * ctx);
```
"""
function gguf_get_n_tensors(ctx)
    ccall((:gguf_get_n_tensors, libllama), Cint, (Ptr{gguf_context},), ctx)
end

"""
    gguf_find_tensor(ctx, name)


### Prototype
```c
int gguf_find_tensor (const struct gguf_context * ctx, const char * name);
```
"""
function gguf_find_tensor(ctx, name)
    ccall((:gguf_find_tensor, libllama), Cint, (Ptr{gguf_context}, Ptr{Cchar}), ctx, name)
end

"""
    gguf_get_tensor_offset(ctx, i)


### Prototype
```c
size_t gguf_get_tensor_offset(const struct gguf_context * ctx, int i);
```
"""
function gguf_get_tensor_offset(ctx, i)
    ccall((:gguf_get_tensor_offset, libllama), Csize_t, (Ptr{gguf_context}, Cint), ctx, i)
end

"""
    gguf_get_tensor_name(ctx, i)


### Prototype
```c
char * gguf_get_tensor_name (const struct gguf_context * ctx, int i);
```
"""
function gguf_get_tensor_name(ctx, i)
    ccall((:gguf_get_tensor_name, libllama), Ptr{Cchar}, (Ptr{gguf_context}, Cint), ctx, i)
end

"""
    gguf_get_tensor_type(ctx, i)


### Prototype
```c
enum ggml_type gguf_get_tensor_type (const struct gguf_context * ctx, int i);
```
"""
function gguf_get_tensor_type(ctx, i)
    ccall((:gguf_get_tensor_type, libllama), ggml_type, (Ptr{gguf_context}, Cint), ctx, i)
end

"""
    gguf_set_val_u8(ctx, key, val)


### Prototype
```c
void gguf_set_val_u8 (struct gguf_context * ctx, const char * key, uint8_t val);
```
"""
function gguf_set_val_u8(ctx, key, val)
    ccall((:gguf_set_val_u8, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, UInt8), ctx, key, val)
end

"""
    gguf_set_val_i8(ctx, key, val)


### Prototype
```c
void gguf_set_val_i8 (struct gguf_context * ctx, const char * key, int8_t val);
```
"""
function gguf_set_val_i8(ctx, key, val)
    ccall((:gguf_set_val_i8, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Int8), ctx, key, val)
end

"""
    gguf_set_val_u16(ctx, key, val)


### Prototype
```c
void gguf_set_val_u16 (struct gguf_context * ctx, const char * key, uint16_t val);
```
"""
function gguf_set_val_u16(ctx, key, val)
    ccall((:gguf_set_val_u16, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, UInt16), ctx, key, val)
end

"""
    gguf_set_val_i16(ctx, key, val)


### Prototype
```c
void gguf_set_val_i16 (struct gguf_context * ctx, const char * key, int16_t val);
```
"""
function gguf_set_val_i16(ctx, key, val)
    ccall((:gguf_set_val_i16, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Int16), ctx, key, val)
end

"""
    gguf_set_val_u32(ctx, key, val)


### Prototype
```c
void gguf_set_val_u32 (struct gguf_context * ctx, const char * key, uint32_t val);
```
"""
function gguf_set_val_u32(ctx, key, val)
    ccall((:gguf_set_val_u32, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, UInt32), ctx, key, val)
end

"""
    gguf_set_val_i32(ctx, key, val)


### Prototype
```c
void gguf_set_val_i32 (struct gguf_context * ctx, const char * key, int32_t val);
```
"""
function gguf_set_val_i32(ctx, key, val)
    ccall((:gguf_set_val_i32, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Int32), ctx, key, val)
end

"""
    gguf_set_val_f32(ctx, key, val)


### Prototype
```c
void gguf_set_val_f32 (struct gguf_context * ctx, const char * key, float val);
```
"""
function gguf_set_val_f32(ctx, key, val)
    ccall((:gguf_set_val_f32, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Cfloat), ctx, key, val)
end

"""
    gguf_set_val_u64(ctx, key, val)


### Prototype
```c
void gguf_set_val_u64 (struct gguf_context * ctx, const char * key, uint64_t val);
```
"""
function gguf_set_val_u64(ctx, key, val)
    ccall((:gguf_set_val_u64, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, UInt64), ctx, key, val)
end

"""
    gguf_set_val_i64(ctx, key, val)


### Prototype
```c
void gguf_set_val_i64 (struct gguf_context * ctx, const char * key, int64_t val);
```
"""
function gguf_set_val_i64(ctx, key, val)
    ccall((:gguf_set_val_i64, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Int64), ctx, key, val)
end

"""
    gguf_set_val_f64(ctx, key, val)


### Prototype
```c
void gguf_set_val_f64 (struct gguf_context * ctx, const char * key, double val);
```
"""
function gguf_set_val_f64(ctx, key, val)
    ccall((:gguf_set_val_f64, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Cdouble), ctx, key, val)
end

"""
    gguf_set_val_bool(ctx, key, val)


### Prototype
```c
void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool val);
```
"""
function gguf_set_val_bool(ctx, key, val)
    ccall((:gguf_set_val_bool, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Bool), ctx, key, val)
end

"""
    gguf_set_val_str(ctx, key, val)


### Prototype
```c
void gguf_set_val_str (struct gguf_context * ctx, const char * key, const char * val);
```
"""
function gguf_set_val_str(ctx, key, val)
    ccall((:gguf_set_val_str, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Ptr{Cchar}), ctx, key, val)
end

"""
    gguf_set_arr_data(ctx, key, type, data, n)


### Prototype
```c
void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n);
```
"""
function gguf_set_arr_data(ctx, key, type, data, n)
    ccall((:gguf_set_arr_data, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, gguf_type, Ptr{Cvoid}, Cint), ctx, key, type, data, n)
end

"""
    gguf_set_arr_str(ctx, key, data, n)


### Prototype
```c
void gguf_set_arr_str (struct gguf_context * ctx, const char * key, const char ** data, int n);
```
"""
function gguf_set_arr_str(ctx, key, data, n)
    ccall((:gguf_set_arr_str, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Ptr{Ptr{Cchar}}, Cint), ctx, key, data, n)
end

"""
    gguf_set_kv(ctx, src)


### Prototype
```c
void gguf_set_kv(struct gguf_context * ctx, struct gguf_context * src);
```
"""
function gguf_set_kv(ctx, src)
    ccall((:gguf_set_kv, libllama), Cvoid, (Ptr{gguf_context}, Ptr{gguf_context}), ctx, src)
end

"""
    gguf_add_tensor(ctx, tensor)


### Prototype
```c
void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor);
```
"""
function gguf_add_tensor(ctx, tensor)
    ccall((:gguf_add_tensor, libllama), Cvoid, (Ptr{gguf_context}, Ptr{ggml_tensor}), ctx, tensor)
end

"""
    gguf_set_tensor_type(ctx, name, type)


### Prototype
```c
void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type);
```
"""
function gguf_set_tensor_type(ctx, name, type)
    ccall((:gguf_set_tensor_type, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, ggml_type), ctx, name, type)
end

"""
    gguf_set_tensor_data(ctx, name, data, size)


### Prototype
```c
void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data, size_t size);
```
"""
function gguf_set_tensor_data(ctx, name, data, size)
    ccall((:gguf_set_tensor_data, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Ptr{Cvoid}, Csize_t), ctx, name, data, size)
end

"""
    gguf_write_to_file(ctx, fname, only_meta)


### Prototype
```c
void gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta);
```
"""
function gguf_write_to_file(ctx, fname, only_meta)
    ccall((:gguf_write_to_file, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Bool), ctx, fname, only_meta)
end

"""
    gguf_get_meta_size(ctx)


### Prototype
```c
size_t gguf_get_meta_size(const struct gguf_context * ctx);
```
"""
function gguf_get_meta_size(ctx)
    ccall((:gguf_get_meta_size, libllama), Csize_t, (Ptr{gguf_context},), ctx)
end

"""
    gguf_get_meta_data(ctx, data)


### Prototype
```c
void gguf_get_meta_data(const struct gguf_context * ctx, void * data);
```
"""
function gguf_get_meta_data(ctx, data)
    ccall((:gguf_get_meta_data, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cvoid}), ctx, data)
end

"""
    ggml_cpu_has_avx()


### Prototype
```c
int ggml_cpu_has_avx (void);
```
"""
function ggml_cpu_has_avx()
    ccall((:ggml_cpu_has_avx, libllama), Cint, ())
end

"""
    ggml_cpu_has_avx_vnni()


### Prototype
```c
int ggml_cpu_has_avx_vnni (void);
```
"""
function ggml_cpu_has_avx_vnni()
    ccall((:ggml_cpu_has_avx_vnni, libllama), Cint, ())
end

"""
    ggml_cpu_has_avx2()


### Prototype
```c
int ggml_cpu_has_avx2 (void);
```
"""
function ggml_cpu_has_avx2()
    ccall((:ggml_cpu_has_avx2, libllama), Cint, ())
end

"""
    ggml_cpu_has_avx512()


### Prototype
```c
int ggml_cpu_has_avx512 (void);
```
"""
function ggml_cpu_has_avx512()
    ccall((:ggml_cpu_has_avx512, libllama), Cint, ())
end

"""
    ggml_cpu_has_avx512_vbmi()


### Prototype
```c
int ggml_cpu_has_avx512_vbmi(void);
```
"""
function ggml_cpu_has_avx512_vbmi()
    ccall((:ggml_cpu_has_avx512_vbmi, libllama), Cint, ())
end

"""
    ggml_cpu_has_avx512_vnni()


### Prototype
```c
int ggml_cpu_has_avx512_vnni(void);
```
"""
function ggml_cpu_has_avx512_vnni()
    ccall((:ggml_cpu_has_avx512_vnni, libllama), Cint, ())
end

"""
    ggml_cpu_has_fma()


### Prototype
```c
int ggml_cpu_has_fma (void);
```
"""
function ggml_cpu_has_fma()
    ccall((:ggml_cpu_has_fma, libllama), Cint, ())
end

"""
    ggml_cpu_has_neon()


### Prototype
```c
int ggml_cpu_has_neon (void);
```
"""
function ggml_cpu_has_neon()
    ccall((:ggml_cpu_has_neon, libllama), Cint, ())
end

"""
    ggml_cpu_has_arm_fma()


### Prototype
```c
int ggml_cpu_has_arm_fma (void);
```
"""
function ggml_cpu_has_arm_fma()
    ccall((:ggml_cpu_has_arm_fma, libllama), Cint, ())
end

"""
    ggml_cpu_has_metal()


### Prototype
```c
int ggml_cpu_has_metal (void);
```
"""
function ggml_cpu_has_metal()
    ccall((:ggml_cpu_has_metal, libllama), Cint, ())
end

"""
    ggml_cpu_has_f16c()


### Prototype
```c
int ggml_cpu_has_f16c (void);
```
"""
function ggml_cpu_has_f16c()
    ccall((:ggml_cpu_has_f16c, libllama), Cint, ())
end

"""
    ggml_cpu_has_fp16_va()


### Prototype
```c
int ggml_cpu_has_fp16_va (void);
```
"""
function ggml_cpu_has_fp16_va()
    ccall((:ggml_cpu_has_fp16_va, libllama), Cint, ())
end

"""
    ggml_cpu_has_wasm_simd()


### Prototype
```c
int ggml_cpu_has_wasm_simd (void);
```
"""
function ggml_cpu_has_wasm_simd()
    ccall((:ggml_cpu_has_wasm_simd, libllama), Cint, ())
end

"""
    ggml_cpu_has_blas()


### Prototype
```c
int ggml_cpu_has_blas (void);
```
"""
function ggml_cpu_has_blas()
    ccall((:ggml_cpu_has_blas, libllama), Cint, ())
end

"""
    ggml_cpu_has_cublas()


### Prototype
```c
int ggml_cpu_has_cublas (void);
```
"""
function ggml_cpu_has_cublas()
    ccall((:ggml_cpu_has_cublas, libllama), Cint, ())
end

"""
    ggml_cpu_has_clblast()


### Prototype
```c
int ggml_cpu_has_clblast (void);
```
"""
function ggml_cpu_has_clblast()
    ccall((:ggml_cpu_has_clblast, libllama), Cint, ())
end

"""
    ggml_cpu_has_gpublas()


### Prototype
```c
int ggml_cpu_has_gpublas (void);
```
"""
function ggml_cpu_has_gpublas()
    ccall((:ggml_cpu_has_gpublas, libllama), Cint, ())
end

"""
    ggml_cpu_has_sse3()


### Prototype
```c
int ggml_cpu_has_sse3 (void);
```
"""
function ggml_cpu_has_sse3()
    ccall((:ggml_cpu_has_sse3, libllama), Cint, ())
end

"""
    ggml_cpu_has_ssse3()


### Prototype
```c
int ggml_cpu_has_ssse3 (void);
```
"""
function ggml_cpu_has_ssse3()
    ccall((:ggml_cpu_has_ssse3, libllama), Cint, ())
end

"""
    ggml_cpu_has_vsx()


### Prototype
```c
int ggml_cpu_has_vsx (void);
```
"""
function ggml_cpu_has_vsx()
    ccall((:ggml_cpu_has_vsx, libllama), Cint, ())
end

# typedef void ( * ggml_to_float_t ) ( const void * GGML_RESTRICT x , float * GGML_RESTRICT y , int k )
const ggml_to_float_t = Ptr{Cvoid}

# typedef void ( * ggml_from_float_t ) ( const float * GGML_RESTRICT x , void * GGML_RESTRICT y , int k )
const ggml_from_float_t = Ptr{Cvoid}

# typedef void ( * ggml_vec_dot_t ) ( const int n , float * GGML_RESTRICT s , const void * GGML_RESTRICT x , const void * GGML_RESTRICT y )
const ggml_vec_dot_t = Ptr{Cvoid}

struct ggml_type_traits_t
    type_name::Ptr{Cchar}
    blck_size::Cint
    type_size::Csize_t
    is_quantized::Bool
    to_float::ggml_to_float_t
    from_float::ggml_from_float_t
    from_float_reference::ggml_from_float_t
    vec_dot::ggml_vec_dot_t
    vec_dot_type::ggml_type
end

"""
    ggml_internal_get_type_traits(type)


### Prototype
```c
ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type type);
```
"""
function ggml_internal_get_type_traits(type)
    ccall((:ggml_internal_get_type_traits, libllama), ggml_type_traits_t, (ggml_type,), type)
end

const LLAMA_MAX_DEVICES = 1

const LLAMA_DEFAULT_SEED = 0xffffffff

const LLAMA_MAX_RNG_STATE = 64 * 1024

const LLAMA_FILE_MAGIC_GGLA = Cuint(0x67676c61)

const LLAMA_FILE_MAGIC_GGSN = Cuint(0x6767736e)

const LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN

const LLAMA_SESSION_VERSION = 3

const GGML_FILE_MAGIC = 0x67676d6c

const GGML_FILE_VERSION = 1

const GGML_QNT_VERSION = 2

const GGML_QNT_VERSION_FACTOR = 1000

const GGML_MAX_DIMS = 4

const GGML_MAX_PARAMS = 2048

const GGML_MAX_CONTEXTS = 64

const GGML_MAX_SRC = 10

const GGML_MAX_NAME = 64

const GGML_MAX_OP_PARAMS = 64

const GGML_DEFAULT_N_THREADS = 4

const GGML_DEFAULT_GRAPH_SIZE = 2048

const GGML_MEM_ALIGN = 16

const GGML_EXIT_SUCCESS = 0

const GGML_EXIT_ABORTED = 1

const GGUF_MAGIC = "GGUF"

const GGUF_VERSION = 3

const GGUF_DEFAULT_ALIGNMENT = 32

const GGML_TENSOR_UNARY_OP_LOCALS = ((((((GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne))(GGML_TENSOR_LOCALS))(size_t, nb0, src0, nb))(GGML_TENSOR_LOCALS))(int64_t, ne, dst, ne))(GGML_TENSOR_LOCALS))(size_t, nb, dst, nb)

const GGML_TENSOR_BINARY_OP_LOCALS = ((((((((((GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne))(GGML_TENSOR_LOCALS))(size_t, nb0, src0, nb))(GGML_TENSOR_LOCALS))(int64_t, ne1, src1, ne))(GGML_TENSOR_LOCALS))(size_t, nb1, src1, nb))(GGML_TENSOR_LOCALS))(int64_t, ne, dst, ne))(GGML_TENSOR_LOCALS))(size_t, nb, dst, nb)

const GGML_N_TASKS_MAX = -1

const GGML_RESTRICT = restrict

# exports
const PREFIXES = ["llama_", "LLAMA_", "ggml_", "GGML_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
