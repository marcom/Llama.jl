using Test
using Llama
using Llama: LibLlama

# show which testset is currently running
showtestset() = println(" "^(2 * Test.get_testset_depth()), "testing ",
                        Test.get_testset().description)

@testset verbose=true "Llama, no model needed" begin
    showtestset()

    @testset "run_llama" begin
        showtestset()
        @test run_llama(; model="", args=`-h`) isa String
        @test run_llama(; model="", prompt="", args=`-h`) isa String
    end

    @testset "run_chat" begin
        showtestset()
        @test run_chat(; model="", args=`-h`) isa Base.Process
    end

    @testset "LlamaContext" begin
        showtestset()
        model_path = "thisfiledoesnotexist.bin"
        @test_throws ErrorException LlamaContext(model_path)

        ctx = LlamaContext()
        @test ctx isa LlamaContext
        @test ctx.token_bos isa LibLlama.llama_token
        @test ctx.token_eos isa LibLlama.llama_token
    end
end

if haskey(ENV, "LLAMA_JL_MODEL_TESTS")
    @info "ENV[\"LLAMA_JL_MODEL_TESTS\"] exists, running tests with model"
    include("tests-with-model.jl")
else
    @info "ENV[\"LLAMA_JL_MODEL_TESTS\"] doesn't exist, _not_ running tests with model"
end
