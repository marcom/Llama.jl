# TODO: make a small dummy weights file
const MODEL_PATH = "../ggml-alpaca-7b-q4.bin"

using Llama: llama_token

@testset "Llama, with model" verbose=true begin
    showtestset()
    @testset "LlamaContext" begin
        showtestset()
        ctx = LlamaContext(MODEL_PATH)
        @test ctx isa LlamaContext
    end

    @testset "tokenize" begin
        showtestset()
        ctx = LlamaContext(MODEL_PATH)
        tokens = tokenize(ctx, "A")
        @test tokens isa Vector{llama_token}
    end
end
