using Test
using Llama

# show which testset is currently running
showtestset() = println(" "^(2 * Test.get_testset_depth()), "testing ",
                        Test.get_testset().description)

@testset verbose=true "Llama" begin
    showtestset()

    @testset "llama" begin
        @test llama(; model="", args=`-h`) isa String
        @test llama(; model="", prompt="", args=`-h`) isa String
    end

    @testset "chat" begin
        @test chat(; model="", args=`-h`) isa Base.Process
    end
end
