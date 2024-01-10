#!/usr/bin/env julia

cd(@__DIR__)
import Pkg
Pkg.activate(".")

using Clang
using Clang.Generators

# Stack environment from parent dir on top of env from this dir.  This
# is so we can import llama_cpp_jll and always have the same version
# as the parent dir.
pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))
println("LOAD_PATH =")
display(LOAD_PATH)
println()

using llama_cpp_jll

options = load_options(joinpath(@__DIR__, "generator.toml"))

include_dir = joinpath(llama_cpp_jll.find_artifact_dir(), "include")

headers = [joinpath(include_dir, "llama.h"), joinpath(include_dir, "ggml.h")]

args = get_default_args()
push!(args, "-I$include_dir")

ctx = create_context(headers, args, options)
build!(ctx)
