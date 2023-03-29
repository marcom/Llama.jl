# TODO: make a small dummy weights file
const MODEL_PATH = "../ggml-alpaca-7b-q4.bin"
const ctx = LlamaContext(MODEL_PATH)

using Llama: llama_token

@testset "Llama, with model" verbose=true begin
    showtestset()
    @testset "LlamaContext" begin
        showtestset()
        @test ctx isa LlamaContext
        @test ctx.n_ctx isa Int
        @test ctx.n_embd isa Int
        @test ctx.n_vocab isa Int
    end

    @testset "tokenize" begin
        showtestset()
        tokens = tokenize(ctx, "A")
        @test tokens isa Vector{llama_token}
    end
end
