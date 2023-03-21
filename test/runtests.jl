using Test
using Llama_cpp

# show which testset is currently running
showtestset() = println(" "^(2 * Test.get_testset_depth()), "testing ",
                        Test.get_testset().description)

@testset verbose=true "Llama_cpp" begin
    showtestset()

    @testset "llama" begin
        @test llama(; model="", prompt="", extra_args=`-h`) isa String
    end
end