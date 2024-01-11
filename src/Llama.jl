module Llama

# Use only these executables for now:
export run_llama, run_chat

# Temporarily unexport as the low-level API is broken!
# export LlamaContext, embeddings, llama_eval, logits, tokenize,
#     token_to_str

import llama_cpp_jll
import ReplMaker

include("../lib/LibLlama.jl")
import .LibLlama

# Temporarily disable as the low-level API is broken!
# __init__() = isdefined(Base, :active_repl) ? init_repl() : nothing

include("api.jl")
include("run-programs.jl")
include("repl.jl")
include("generate.jl")

end # module Llama
