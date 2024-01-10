using Test
using Llama
using Llama: LibLlama

# show which testset is currently running
function showtestset()
    println(" "^(2 * Test.get_testset_depth()), "testing ",
        Test.get_testset().description)
end

include("run-programs.jl")
