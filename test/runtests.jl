using Test
using Aqua
using LlamaCpp
using LlamaCpp: LibLlama

# show which testset is currently running
function showtestset()
    println(" "^(2 * Test.get_testset_depth()), "testing ",
        Test.get_testset().description)
end

@testset "Code quality (Aqua.jl)" begin
    # Skipping unbound_args check because we need our `MaybeExtract` type to be unboard
    Aqua.test_all(LlamaCpp)
end

@testset "LlamaCpp.jl" begin
    include("utils.jl")
    include("run-programs.jl")
end
