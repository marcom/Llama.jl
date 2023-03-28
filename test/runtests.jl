using Test
using Llama

# show which testset is currently running
showtestset() = println(" "^(2 * Test.get_testset_depth()), "testing ",
                        Test.get_testset().description)

@testset verbose=true "Llama" begin
    showtestset()

    @testset "run_llama" begin
        @test run_llama(; model="", args=`-h`) isa String
        @test run_llama(; model="", prompt="", args=`-h`) isa String
    end

    @testset "run_chat" begin
        @test run_chat(; model="", args=`-h`) isa Base.Process
    end

    @testset "LlamaContext" begin
        model_path = "thisfiledoesnotexist.bin"
        @test_throws ErrorException LlamaContext(model_path)

        # TODO: commented out for now
        # - could be tested if we can make a small dummy weights file
        #model_path = "ggml-alpaca-7b-q4.bin"
        #@test LlamaContext(model_path) isa LlamaContext
    end
end
