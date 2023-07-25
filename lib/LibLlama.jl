module LibLlama

using llama_cpp_jll
export llama_cpp_jll

using CEnum

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
    GGML_TYPE_I8 = 16
    GGML_TYPE_I16 = 17
    GGML_TYPE_I32 = 18
    GGML_TYPE_COUNT = 19
end

@cenum ggml_backend::UInt32 begin
    GGML_BACKEND_CPU = 0
    GGML_BACKEND_GPU = 10
    GGML_BACKEND_GPU_SPLIT = 20
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
end

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
    GGML_OP_SILU_BACK = 17
    GGML_OP_NORM = 18
    GGML_OP_RMS_NORM = 19
    GGML_OP_RMS_NORM_BACK = 20
    GGML_OP_MUL_MAT = 21
    GGML_OP_OUT_PROD = 22
    GGML_OP_SCALE = 23
    GGML_OP_SET = 24
    GGML_OP_CPY = 25
    GGML_OP_CONT = 26
    GGML_OP_RESHAPE = 27
    GGML_OP_VIEW = 28
    GGML_OP_PERMUTE = 29
    GGML_OP_TRANSPOSE = 30
    GGML_OP_GET_ROWS = 31
    GGML_OP_GET_ROWS_BACK = 32
    GGML_OP_DIAG = 33
    GGML_OP_DIAG_MASK_INF = 34
    GGML_OP_DIAG_MASK_ZERO = 35
    GGML_OP_SOFT_MAX = 36
    GGML_OP_SOFT_MAX_BACK = 37
    GGML_OP_ROPE = 38
    GGML_OP_ROPE_BACK = 39
    GGML_OP_ALIBI = 40
    GGML_OP_CLAMP = 41
    GGML_OP_CONV_1D = 42
    GGML_OP_CONV_2D = 43
    GGML_OP_POOL_1D = 44
    GGML_OP_POOL_2D = 45
    GGML_OP_FLASH_ATTN = 46
    GGML_OP_FLASH_FF = 47
    GGML_OP_FLASH_ATTN_BACK = 48
    GGML_OP_WIN_PART = 49
    GGML_OP_WIN_UNPART = 50
    GGML_OP_UNARY = 51
    GGML_OP_MAP_UNARY = 52
    GGML_OP_MAP_BINARY = 53
    GGML_OP_MAP_CUSTOM1 = 54
    GGML_OP_MAP_CUSTOM2 = 55
    GGML_OP_MAP_CUSTOM3 = 56
    GGML_OP_CROSS_ENTROPY_LOSS = 57
    GGML_OP_CROSS_ENTROPY_LOSS_BACK = 58
    GGML_OP_COUNT = 59
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
end

struct ggml_object
    offs::Csize_t
    size::Csize_t
    next::Ptr{ggml_object}
    padding::NTuple{8, Cchar}
end

struct ggml_tensor
    type::ggml_type
    backend::ggml_backend
    n_dims::Cint
    ne::NTuple{4, Int64} # ne::NTuple{4, Int64}
    nb::NTuple{4, Csize_t} # nb::NTuple{4, Csize_t}
    op::ggml_op
    op_params::NTuple{8, Int32} # op_params::NTuple{8, Int32}
    is_param::Bool
    grad::Ptr{ggml_tensor}
    src::NTuple{6, Ptr{ggml_tensor}}
    perf_runs::Cint
    perf_cycles::Int64
    perf_time_us::Int64
    data::Ptr{Cvoid}
    name::NTuple{48, Cchar}
    extra::Ptr{Cvoid}
    padding::NTuple{8, Cchar}
end

function Base.getproperty(x::ggml_tensor, f::Symbol)
    f === :ne && return NTuple{4, Int64}(getfield(x, f))
    f === :nb && return NTuple{4, Csize_t}(getfield(x, f))
    f === :op_params && return NTuple{8, Int32}(getfield(x, f))
    return getfield(x, f)
end

struct ggml_cplan
    work_size::Csize_t
    work_data::Ptr{UInt8}
    n_threads::Cint
    n_tasks::NTuple{4096, Cint}
    abort_callback::Ptr{Cvoid}
    abort_callback_data::Ptr{Cvoid}
end

struct ggml_cgraph
    n_nodes::Cint
    n_leafs::Cint
    nodes::NTuple{4096, Ptr{ggml_tensor}}
    grads::NTuple{4096, Ptr{ggml_tensor}}
    leafs::NTuple{4096, Ptr{ggml_tensor}}
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
    ggml_nbytes_split(tensor, nrows_split)


### Prototype
```c
size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split);
```
"""
function ggml_nbytes_split(tensor, nrows_split)
    ccall((:ggml_nbytes_split, libllama), Csize_t, (Ptr{ggml_tensor}, Cint), tensor, nrows_split)
end

"""
    ggml_blck_size(type)


### Prototype
```c
int ggml_blck_size (enum ggml_type type);
```
"""
function ggml_blck_size(type)
    ccall((:ggml_blck_size, libllama), Cint, (ggml_type,), type)
end

"""
    ggml_type_size(type)


### Prototype
```c
size_t ggml_type_size (enum ggml_type type);
```
"""
function ggml_type_size(type)
    ccall((:ggml_type_size, libllama), Csize_t, (ggml_type,), type)
end

"""
    ggml_type_sizef(type)


### Prototype
```c
float ggml_type_sizef(enum ggml_type type);
```
"""
function ggml_type_sizef(type)
    ccall((:ggml_type_sizef, libllama), Cfloat, (ggml_type,), type)
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
struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, const struct ggml_tensor * src);
```
"""
function ggml_view_tensor(ctx, src)
    ccall((:ggml_view_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, src)
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
    ggml_norm(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_norm( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_norm(ctx, a)
    ccall((:ggml_norm, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

"""
    ggml_norm_inplace(ctx, a)


### Prototype
```c
struct ggml_tensor * ggml_norm_inplace( struct ggml_context * ctx, struct ggml_tensor * a);
```
"""
function ggml_norm_inplace(ctx, a)
    ccall((:ggml_norm_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
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
    ggml_rms_norm_back(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_rms_norm_back( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_rms_norm_back(ctx, a, b)
    ccall((:ggml_rms_norm_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
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
    ggml_scale(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_scale( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_scale(ctx, a, b)
    ccall((:ggml_scale, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

"""
    ggml_scale_inplace(ctx, a, b)


### Prototype
```c
struct ggml_tensor * ggml_scale_inplace( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```
"""
function ggml_scale_inplace(ctx, a, b)
    ccall((:ggml_scale_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
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
    ggml_rope(ctx, a, n_past, n_dims, mode, n_ctx)


### Prototype
```c
struct ggml_tensor * ggml_rope( struct ggml_context * ctx, struct ggml_tensor * a, int n_past, int n_dims, int mode, int n_ctx);
```
"""
function ggml_rope(ctx, a, n_past, n_dims, mode, n_ctx)
    ccall((:ggml_rope, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, n_past, n_dims, mode, n_ctx)
end

"""
    ggml_rope_inplace(ctx, a, n_past, n_dims, mode, n_ctx)


### Prototype
```c
struct ggml_tensor * ggml_rope_inplace( struct ggml_context * ctx, struct ggml_tensor * a, int n_past, int n_dims, int mode, int n_ctx);
```
"""
function ggml_rope_inplace(ctx, a, n_past, n_dims, mode, n_ctx)
    ccall((:ggml_rope_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, n_past, n_dims, mode, n_ctx)
end

"""
    ggml_rope_custom_inplace(ctx, a, n_past, n_dims, mode, n_ctx, freq_base, freq_scale)


### Prototype
```c
struct ggml_tensor * ggml_rope_custom_inplace( struct ggml_context * ctx, struct ggml_tensor * a, int n_past, int n_dims, int mode, int n_ctx, float freq_base, float freq_scale);
```
"""
function ggml_rope_custom_inplace(ctx, a, n_past, n_dims, mode, n_ctx, freq_base, freq_scale)
    ccall((:ggml_rope_custom_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cfloat, Cfloat), ctx, a, n_past, n_dims, mode, n_ctx, freq_base, freq_scale)
end

"""
    ggml_rope_back(ctx, a, n_past, n_dims, mode, n_ctx)


### Prototype
```c
struct ggml_tensor * ggml_rope_back( struct ggml_context * ctx, struct ggml_tensor * a, int n_past, int n_dims, int mode, int n_ctx);
```
"""
function ggml_rope_back(ctx, a, n_past, n_dims, mode, n_ctx)
    ccall((:ggml_rope_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, n_past, n_dims, mode, n_ctx)
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
    ggml_conv_1d_ph(ctx, a, b, s, d)


### Prototype
```c
struct ggml_tensor* ggml_conv_1d_ph( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s, int d);
```
"""
function ggml_conv_1d_ph(ctx, a, b, s, d)
    ccall((:ggml_conv_1d_ph, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, b, s, d)
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
struct ggml_tensor* ggml_pool_1d( struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_op_pool op, int k0, int s0, int p0);
```
"""
function ggml_pool_1d(ctx, a, op, k0, s0, p0)
    ccall((:ggml_pool_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_op_pool, Cint, Cint, Cint), ctx, a, op, k0, s0, p0)
end

"""
    ggml_pool_2d(ctx, a, op, k0, k1, s0, s1, p0, p1)


### Prototype
```c
struct ggml_tensor* ggml_pool_2d( struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_op_pool op, int k0, int k1, int s0, int s1, int p0, int p1);
```
"""
function ggml_pool_2d(ctx, a, op, k0, k1, s0, s1, p0, p1)
    ccall((:ggml_pool_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_op_pool, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, op, k0, k1, s0, s1, p0, p1)
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
    ggml_map_unary_f32(ctx, a, fun)


### Prototype
```c
struct ggml_tensor * ggml_map_unary_f32( struct ggml_context * ctx, struct ggml_tensor * a, ggml_unary_op_f32_t fun);
```
"""
function ggml_map_unary_f32(ctx, a, fun)
    ccall((:ggml_map_unary_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_unary_op_f32_t), ctx, a, fun)
end

"""
    ggml_map_unary_inplace_f32(ctx, a, fun)


### Prototype
```c
struct ggml_tensor * ggml_map_unary_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, ggml_unary_op_f32_t fun);
```
"""
function ggml_map_unary_inplace_f32(ctx, a, fun)
    ccall((:ggml_map_unary_inplace_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_unary_op_f32_t), ctx, a, fun)
end

"""
    ggml_map_binary_f32(ctx, a, b, fun)


### Prototype
```c
struct ggml_tensor * ggml_map_binary_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_binary_op_f32_t fun);
```
"""
function ggml_map_binary_f32(ctx, a, b, fun)
    ccall((:ggml_map_binary_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_binary_op_f32_t), ctx, a, b, fun)
end

"""
    ggml_map_binary_inplace_f32(ctx, a, b, fun)


### Prototype
```c
struct ggml_tensor * ggml_map_binary_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_binary_op_f32_t fun);
```
"""
function ggml_map_binary_inplace_f32(ctx, a, b, fun)
    ccall((:ggml_map_binary_inplace_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_binary_op_f32_t), ctx, a, b, fun)
end

"""
    ggml_map_custom1_f32(ctx, a, fun)


### Prototype
```c
struct ggml_tensor * ggml_map_custom1_f32( struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_f32_t fun);
```
"""
function ggml_map_custom1_f32(ctx, a, fun)
    ccall((:ggml_map_custom1_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_custom1_op_f32_t), ctx, a, fun)
end

"""
    ggml_map_custom1_inplace_f32(ctx, a, fun)


### Prototype
```c
struct ggml_tensor * ggml_map_custom1_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_f32_t fun);
```
"""
function ggml_map_custom1_inplace_f32(ctx, a, fun)
    ccall((:ggml_map_custom1_inplace_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_custom1_op_f32_t), ctx, a, fun)
end

"""
    ggml_map_custom2_f32(ctx, a, b, fun)


### Prototype
```c
struct ggml_tensor * ggml_map_custom2_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_f32_t fun);
```
"""
function ggml_map_custom2_f32(ctx, a, b, fun)
    ccall((:ggml_map_custom2_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom2_op_f32_t), ctx, a, b, fun)
end

"""
    ggml_map_custom2_inplace_f32(ctx, a, b, fun)


### Prototype
```c
struct ggml_tensor * ggml_map_custom2_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_f32_t fun);
```
"""
function ggml_map_custom2_inplace_f32(ctx, a, b, fun)
    ccall((:ggml_map_custom2_inplace_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom2_op_f32_t), ctx, a, b, fun)
end

"""
    ggml_map_custom3_f32(ctx, a, b, c, fun)


### Prototype
```c
struct ggml_tensor * ggml_map_custom3_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_f32_t fun);
```
"""
function ggml_map_custom3_f32(ctx, a, b, c, fun)
    ccall((:ggml_map_custom3_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom3_op_f32_t), ctx, a, b, c, fun)
end

"""
    ggml_map_custom3_inplace_f32(ctx, a, b, c, fun)


### Prototype
```c
struct ggml_tensor * ggml_map_custom3_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_f32_t fun);
```
"""
function ggml_map_custom3_inplace_f32(ctx, a, b, c, fun)
    ccall((:ggml_map_custom3_inplace_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom3_op_f32_t), ctx, a, b, c, fun)
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
void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
```
"""
function ggml_build_forward_expand(cgraph, tensor)
    ccall((:ggml_build_forward_expand, libllama), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_tensor}), cgraph, tensor)
end

"""
    ggml_build_forward(tensor)


### Prototype
```c
struct ggml_cgraph ggml_build_forward (struct ggml_tensor * tensor);
```
"""
function ggml_build_forward(tensor)
    ccall((:ggml_build_forward, libllama), ggml_cgraph, (Ptr{ggml_tensor},), tensor)
end

"""
    ggml_build_backward(ctx, gf, keep)


### Prototype
```c
struct ggml_cgraph ggml_build_backward(struct ggml_context * ctx, struct ggml_cgraph * gf, bool keep);
```
"""
function ggml_build_backward(ctx, gf, keep)
    ccall((:ggml_build_backward, libllama), ggml_cgraph, (Ptr{ggml_context}, Ptr{ggml_cgraph}, Bool), ctx, gf, keep)
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
struct ggml_cgraph ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval);
```
"""
function ggml_graph_import(fname, ctx_data, ctx_eval)
    ccall((:ggml_graph_import, libllama), ggml_cgraph, (Ptr{Cchar}, Ptr{Ptr{ggml_context}}, Ptr{Ptr{ggml_context}}), fname, ctx_data, ctx_eval)
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
    GGML_LINESEARCH_FAIL = -128
    GGML_LINESEARCH_MINIMUM_STEP = -127
    GGML_LINESEARCH_MAXIMUM_STEP = -126
    GGML_LINESEARCH_MAXIMUM_ITERATIONS = -125
    GGML_LINESEARCH_INVALID_PARAMETERS = -124
end

struct __JL_Ctag_5
    n_iter::Cint
    sched::Cfloat
    decay::Cfloat
    alpha::Cfloat
    beta1::Cfloat
    beta2::Cfloat
    eps::Cfloat
    eps_f::Cfloat
    eps_g::Cfloat
end
function Base.getproperty(x::Ptr{__JL_Ctag_5}, f::Symbol)
    f === :n_iter && return Ptr{Cint}(x + 0)
    f === :sched && return Ptr{Cfloat}(x + 4)
    f === :decay && return Ptr{Cfloat}(x + 8)
    f === :alpha && return Ptr{Cfloat}(x + 12)
    f === :beta1 && return Ptr{Cfloat}(x + 16)
    f === :beta2 && return Ptr{Cfloat}(x + 20)
    f === :eps && return Ptr{Cfloat}(x + 24)
    f === :eps_f && return Ptr{Cfloat}(x + 28)
    f === :eps_g && return Ptr{Cfloat}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_5, f::Symbol)
    r = Ref{__JL_Ctag_5}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_5}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_5}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct __JL_Ctag_6
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
function Base.getproperty(x::Ptr{__JL_Ctag_6}, f::Symbol)
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

function Base.getproperty(x::__JL_Ctag_6, f::Symbol)
    r = Ref{__JL_Ctag_6}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_6}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_6}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct ggml_opt_params
    data::NTuple{96, UInt8}
end

function Base.getproperty(x::Ptr{ggml_opt_params}, f::Symbol)
    f === :type && return Ptr{ggml_opt_type}(x + 0)
    f === :n_threads && return Ptr{Cint}(x + 4)
    f === :past && return Ptr{Cint}(x + 8)
    f === :delta && return Ptr{Cfloat}(x + 12)
    f === :max_no_improvement && return Ptr{Cint}(x + 16)
    f === :print_forward_graph && return Ptr{Bool}(x + 20)
    f === :print_backward_graph && return Ptr{Bool}(x + 21)
    f === :adam && return Ptr{__JL_Ctag_5}(x + 24)
    f === :lbfgs && return Ptr{__JL_Ctag_6}(x + 60)
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

struct __JL_Ctag_3
    x::Ptr{ggml_tensor}
    g1::Ptr{ggml_tensor}
    g2::Ptr{ggml_tensor}
    m::Ptr{ggml_tensor}
    v::Ptr{ggml_tensor}
    mh::Ptr{ggml_tensor}
    vh::Ptr{ggml_tensor}
    pf::Ptr{ggml_tensor}
    fx_best::Cfloat
    fx_prev::Cfloat
    n_no_improvement::Cint
end
function Base.getproperty(x::Ptr{__JL_Ctag_3}, f::Symbol)
    f === :x && return Ptr{Ptr{ggml_tensor}}(x + 0)
    f === :g1 && return Ptr{Ptr{ggml_tensor}}(x + 8)
    f === :g2 && return Ptr{Ptr{ggml_tensor}}(x + 16)
    f === :m && return Ptr{Ptr{ggml_tensor}}(x + 24)
    f === :v && return Ptr{Ptr{ggml_tensor}}(x + 32)
    f === :mh && return Ptr{Ptr{ggml_tensor}}(x + 40)
    f === :vh && return Ptr{Ptr{ggml_tensor}}(x + 48)
    f === :pf && return Ptr{Ptr{ggml_tensor}}(x + 56)
    f === :fx_best && return Ptr{Cfloat}(x + 64)
    f === :fx_prev && return Ptr{Cfloat}(x + 68)
    f === :n_no_improvement && return Ptr{Cint}(x + 72)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_3, f::Symbol)
    r = Ref{__JL_Ctag_3}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_3}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_3}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct __JL_Ctag_4
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
function Base.getproperty(x::Ptr{__JL_Ctag_4}, f::Symbol)
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

function Base.getproperty(x::__JL_Ctag_4, f::Symbol)
    r = Ref{__JL_Ctag_4}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_4}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_4}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct ggml_opt_context
    data::NTuple{312, UInt8}
end

function Base.getproperty(x::Ptr{ggml_opt_context}, f::Symbol)
    f === :ctx && return Ptr{Ptr{ggml_context}}(x + 0)
    f === :params && return Ptr{ggml_opt_params}(x + 8)
    f === :iter && return Ptr{Cint}(x + 104)
    f === :nx && return Ptr{Int64}(x + 112)
    f === :just_initialized && return Ptr{Bool}(x + 120)
    f === :adam && return Ptr{__JL_Ctag_3}(x + 128)
    f === :lbfgs && return Ptr{__JL_Ctag_4}(x + 208)
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
    ggml_opt_resume_g(ctx, opt, f, gf, gb)


### Prototype
```c
enum ggml_opt_result ggml_opt_resume_g( struct ggml_context * ctx, struct ggml_opt_context * opt, struct ggml_tensor * f, struct ggml_cgraph * gf, struct ggml_cgraph * gb);
```
"""
function ggml_opt_resume_g(ctx, opt, f, gf, gb)
    ccall((:ggml_opt_resume_g, libllama), ggml_opt_result, (Ptr{ggml_context}, Ptr{ggml_opt_context}, Ptr{ggml_tensor}, Ptr{ggml_cgraph}, Ptr{ggml_cgraph}), ctx, opt, f, gf, gb)
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
    ggml_quantize_chunk(type, src, dst, start, n, hist)


### Prototype
```c
size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst, int start, int n, int64_t * hist);
```
"""
function ggml_quantize_chunk(type, src, dst, start, n, hist)
    ccall((:ggml_quantize_chunk, libllama), Csize_t, (ggml_type, Ptr{Cfloat}, Ptr{Cvoid}, Cint, Cint, Ptr{Int64}), type, src, dst, start, n, hist)
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
    to_float::ggml_to_float_t
    from_float::ggml_from_float_t
    from_float_reference::ggml_from_float_t
    vec_dot::ggml_vec_dot_t
    vec_dot_type::ggml_type
end

"""
    ggml_internal_get_type_traits(i)


### Prototype
```c
ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type i);
```
"""
function ggml_internal_get_type_traits(i)
    ccall((:ggml_internal_get_type_traits, libllama), ggml_type_traits_t, (ggml_type,), i)
end

mutable struct llama_model end

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
    seed::UInt32
    n_ctx::Int32
    n_batch::Int32
    n_gqa::Int32
    rms_norm_eps::Cfloat
    n_gpu_layers::Int32
    main_gpu::Int32
    tensor_split::Ptr{Cfloat}
    rope_freq_base::Cfloat
    rope_freq_scale::Cfloat
    progress_callback::llama_progress_callback
    progress_callback_user_data::Ptr{Cvoid}
    low_vram::Bool
    f16_kv::Bool
    logits_all::Bool
    vocab_only::Bool
    use_mmap::Bool
    use_mlock::Bool
    embedding::Bool
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
end

struct llama_model_quantize_params
    nthread::Cint
    ftype::llama_ftype
    allow_requantize::Bool
    quantize_output_tensor::Bool
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

# no prototype is found for this function at llama.h:193:19, please use with caution
"""
    llama_max_devices()


### Prototype
```c
int llama_max_devices();
```
"""
function llama_max_devices()
    ccall((:llama_max_devices, libllama), Cint, ())
end

# no prototype is found for this function at llama.h:195:43, please use with caution
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

# no prototype is found for this function at llama.h:196:50, please use with caution
"""
    llama_model_quantize_default_params()


### Prototype
```c
struct llama_model_quantize_params llama_model_quantize_default_params();
```
"""
function llama_model_quantize_default_params()
    ccall((:llama_model_quantize_default_params, libllama), llama_model_quantize_params, ())
end

# no prototype is found for this function at llama.h:198:20, please use with caution
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

# no prototype is found for this function at llama.h:199:20, please use with caution
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
    llama_backend_init(numa)


### Prototype
```c
void llama_backend_init(bool numa);
```
"""
function llama_backend_init(numa)
    ccall((:llama_backend_init, libllama), Cvoid, (Bool,), numa)
end

# no prototype is found for this function at llama.h:207:20, please use with caution
"""
    llama_backend_free()


### Prototype
```c
void llama_backend_free();
```
"""
function llama_backend_free()
    ccall((:llama_backend_free, libllama), Cvoid, ())
end

# no prototype is found for this function at llama.h:209:23, please use with caution
"""
    llama_time_us()


### Prototype
```c
int64_t llama_time_us();
```
"""
function llama_time_us()
    ccall((:llama_time_us, libllama), Int64, ())
end

"""
    llama_load_model_from_file(path_model, params)


### Prototype
```c
struct llama_model * llama_load_model_from_file( const char * path_model, struct llama_context_params params);
```
"""
function llama_load_model_from_file(path_model, params)
    ccall((:llama_load_model_from_file, libllama), Ptr{llama_model}, (Ptr{Cchar}, llama_context_params), path_model, params)
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
    llama_init_from_file(path_model, params)


### Prototype
```c
DEPRECATED(struct llama_context * llama_init_from_file( const char * path_model, struct llama_context_params params), "please use llama_load_model_from_file combined with llama_new_context_with_model instead");
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
    llama_model_quantize(fname_inp, fname_out, params)


### Prototype
```c
int llama_model_quantize( const char * fname_inp, const char * fname_out, const llama_model_quantize_params * params);
```
"""
function llama_model_quantize(fname_inp, fname_out, params)
    ccall((:llama_model_quantize, libllama), Cint, (Ptr{Cchar}, Ptr{Cchar}, Ptr{llama_model_quantize_params}), fname_inp, fname_out, params)
end

"""
    llama_apply_lora_from_file(ctx, path_lora, path_base_model, n_threads)


### Prototype
```c
DEPRECATED(int llama_apply_lora_from_file( struct llama_context * ctx, const char * path_lora, const char * path_base_model, int n_threads), "please use llama_model_apply_lora_from_file instead");
```
"""
function llama_apply_lora_from_file(ctx, path_lora, path_base_model, n_threads)
    ccall((:llama_apply_lora_from_file, libllama), Cint, (Ptr{llama_context}, Ptr{Cchar}, Ptr{Cchar}, Cint), ctx, path_lora, path_base_model, n_threads)
end

"""
    llama_model_apply_lora_from_file(model, path_lora, path_base_model, n_threads)


### Prototype
```c
int llama_model_apply_lora_from_file( const struct llama_model * model, const char * path_lora, const char * path_base_model, int n_threads);
```
"""
function llama_model_apply_lora_from_file(model, path_lora, path_base_model, n_threads)
    ccall((:llama_model_apply_lora_from_file, libllama), Cint, (Ptr{llama_model}, Ptr{Cchar}, Ptr{Cchar}, Cint), model, path_lora, path_base_model, n_threads)
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
void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed);
```
"""
function llama_set_rng_seed(ctx, seed)
    ccall((:llama_set_rng_seed, libllama), Cvoid, (Ptr{llama_context}, UInt32), ctx, seed)
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
size_t llama_set_state_data(struct llama_context * ctx, uint8_t * src);
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
    llama_eval_embd(ctx, embd, n_tokens, n_past, n_threads)


### Prototype
```c
int llama_eval_embd( struct llama_context * ctx, const float * embd, int n_tokens, int n_past, int n_threads);
```
"""
function llama_eval_embd(ctx, embd, n_tokens, n_past, n_threads)
    ccall((:llama_eval_embd, libllama), Cint, (Ptr{llama_context}, Ptr{Cfloat}, Cint, Cint, Cint), ctx, embd, n_tokens, n_past, n_threads)
end

"""
    llama_eval_export(ctx, fname)


### Prototype
```c
int llama_eval_export(struct llama_context * ctx, const char * fname);
```
"""
function llama_eval_export(ctx, fname)
    ccall((:llama_eval_export, libllama), Cint, (Ptr{llama_context}, Ptr{Cchar}), ctx, fname)
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
    llama_tokenize_with_model(model, text, tokens, n_max_tokens, add_bos)


### Prototype
```c
int llama_tokenize_with_model( const struct llama_model * model, const char * text, llama_token * tokens, int n_max_tokens, bool add_bos);
```
"""
function llama_tokenize_with_model(model, text, tokens, n_max_tokens, add_bos)
    ccall((:llama_tokenize_with_model, libllama), Cint, (Ptr{llama_model}, Ptr{Cchar}, Ptr{llama_token}, Cint, Bool), model, text, tokens, n_max_tokens, add_bos)
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
    llama_n_vocab_from_model(model)


### Prototype
```c
int llama_n_vocab_from_model(const struct llama_model * model);
```
"""
function llama_n_vocab_from_model(model)
    ccall((:llama_n_vocab_from_model, libllama), Cint, (Ptr{llama_model},), model)
end

"""
    llama_n_ctx_from_model(model)


### Prototype
```c
int llama_n_ctx_from_model (const struct llama_model * model);
```
"""
function llama_n_ctx_from_model(model)
    ccall((:llama_n_ctx_from_model, libllama), Cint, (Ptr{llama_model},), model)
end

"""
    llama_n_embd_from_model(model)


### Prototype
```c
int llama_n_embd_from_model (const struct llama_model * model);
```
"""
function llama_n_embd_from_model(model)
    ccall((:llama_n_embd_from_model, libllama), Cint, (Ptr{llama_model},), model)
end

"""
    llama_get_vocab(ctx, strings, scores, capacity)


### Prototype
```c
int llama_get_vocab( const struct llama_context * ctx, const char * * strings, float * scores, int capacity);
```
"""
function llama_get_vocab(ctx, strings, scores, capacity)
    ccall((:llama_get_vocab, libllama), Cint, (Ptr{llama_context}, Ptr{Ptr{Cchar}}, Ptr{Cfloat}, Cint), ctx, strings, scores, capacity)
end

"""
    llama_get_vocab_from_model(model, strings, scores, capacity)


### Prototype
```c
int llama_get_vocab_from_model( const struct llama_model * model, const char * * strings, float * scores, int capacity);
```
"""
function llama_get_vocab_from_model(model, strings, scores, capacity)
    ccall((:llama_get_vocab_from_model, libllama), Cint, (Ptr{llama_model}, Ptr{Ptr{Cchar}}, Ptr{Cfloat}, Cint), model, strings, scores, capacity)
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
const char * llama_token_to_str( const struct llama_context * ctx, llama_token token);
```
"""
function llama_token_to_str(ctx, token)
    ccall((:llama_token_to_str, libllama), Ptr{Cchar}, (Ptr{llama_context}, llama_token), ctx, token)
end

"""
    llama_token_to_str_with_model(model, token)


### Prototype
```c
const char * llama_token_to_str_with_model( const struct llama_model * model, llama_token token);
```
"""
function llama_token_to_str_with_model(model, token)
    ccall((:llama_token_to_str_with_model, libllama), Ptr{Cchar}, (Ptr{llama_model}, llama_token), model, token)
end

# no prototype is found for this function at llama.h:367:27, please use with caution
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

# no prototype is found for this function at llama.h:368:27, please use with caution
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

# no prototype is found for this function at llama.h:369:27, please use with caution
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
    llama_sample_grammar(ctx, candidates, grammar)

@details Apply constraints from grammar
### Prototype
```c
void llama_sample_grammar(struct llama_context * ctx, llama_token_data_array * candidates, const struct llama_grammar * grammar);
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
    llama_grammar_accept_token(ctx, grammar, token)

@details Accepts the sampled token into the grammar
### Prototype
```c
void llama_grammar_accept_token(struct llama_context * ctx, struct llama_grammar * grammar, llama_token token);
```
"""
function llama_grammar_accept_token(ctx, grammar, token)
    ccall((:llama_grammar_accept_token, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_grammar}, llama_token), ctx, grammar, token)
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

const GGML_FILE_MAGIC = 0x67676d6c

const GGML_FILE_VERSION = 1

const GGML_QNT_VERSION = 2

const GGML_QNT_VERSION_FACTOR = 1000

const GGML_MAX_DIMS = 4

const GGML_MAX_NODES = 4096

const GGML_MAX_PARAMS = 256

const GGML_MAX_CONTEXTS = 64

const GGML_MAX_SRC = 6

const GGML_MAX_NAME = 48

const GGML_MAX_OP_PARAMS = 32

const GGML_DEFAULT_N_THREADS = 4

const GGML_EXIT_SUCCESS = 0

const GGML_EXIT_ABORTED = 1

# const GGML_RESTRICT = restrict

const LLAMA_MAX_DEVICES = 1

const LLAMA_FILE_MAGIC_GGJT = Cuint(0x67676a74)

const LLAMA_FILE_MAGIC_GGLA = Cuint(0x67676c61)

const LLAMA_FILE_MAGIC_GGMF = Cuint(0x67676d66)

const LLAMA_FILE_MAGIC_GGML = Cuint(0x67676d6c)

const LLAMA_FILE_MAGIC_GGSN = Cuint(0x6767736e)

const LLAMA_FILE_VERSION = 3

const LLAMA_FILE_MAGIC = LLAMA_FILE_MAGIC_GGJT

const LLAMA_FILE_MAGIC_UNVERSIONED = LLAMA_FILE_MAGIC_GGML

const LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN

const LLAMA_SESSION_VERSION = 1

const LLAMA_DEFAULT_SEED = 0xffffffff

# exports
const PREFIXES = ["llama_", "LLAMA_", "ggml_", "GGML_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
