using Clang
using Clang.Generators

using llama_cpp_jll

d = pwd()
cd(@__DIR__)

options = load_options(joinpath(@__DIR__, "generator.toml"))


include_dir = joinpath(llama_cpp_jll.find_artifact_dir(), "include")

# doesn't work yet, assertion error in Clang.jl
#headers = [joinpath(include_dir, "llama.h"), joinpath(include_dir, "ggml.h")]
headers = [joinpath(include_dir, "llama.h")]

args = get_default_args() 
push!(args, "-I$include_dir")

ctx = create_context(headers, args, options)
build!(ctx)

cd(d)
