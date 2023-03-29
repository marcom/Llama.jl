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

    @testset "logits" begin
        showtestset()
        lg = logits(ctx)
        @test lg isa Vector{Float32}
        @test length(lg) == ctx.n_vocab
    end

    @testset "tokenize" begin
        showtestset()
        tokens = tokenize(ctx, "A")
        @test tokens isa Vector{llama_token}
    end

    @testset "token_to_str" begin
        showtestset()
        str = token_to_str(ctx, 100)
        @test str isa String
    end
end
