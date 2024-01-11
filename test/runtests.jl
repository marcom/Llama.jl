using Test
using Aqua
using Llama
using Llama: LibLlama

# show which testset is currently running
showtestset() = println(" "^(2 * Test.get_testset_depth()), "testing ",
    Test.get_testset().description)

@testset "Code quality (Aqua.jl)" begin
    # Skipping unbound_args check because we need our `MaybeExtract` type to be unboard
    Aqua.test_all(Llama)
end

@testset "Llama.jl" begin
    include("utils.jl")
    include("run-programs.jl")
end
