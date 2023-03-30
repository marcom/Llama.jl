# TODO: make a small dummy weights file
const MODEL_PATH = "../ggml-alpaca-7b-q4.bin"
const ctx = LlamaContext(MODEL_PATH)

using Llama: LibLlama

@testset "Llama, with model" verbose=true begin
    showtestset()
    @testset "LlamaContext" begin
        showtestset()
        @test ctx isa LlamaContext
        @test ctx.n_ctx isa Int
        @test ctx.n_embd isa Int
        @test ctx.n_vocab isa Int
    end

    @testset "embeddings" begin
        showtestset()
        # TODO: emebddings require the net to have been run with llama_eval
        #em = embeddings(ctx)
        #@test em isa Vector{Float32}
        #@test length(em) == ctx.n_embd
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
        @test tokens isa Vector{LibLlama.llama_token}
    end

    @testset "llama_eval" begin
        showtestset()
        tokens = LibLlama.llama_token[10, 11, 12, 13]
        n_past = 0
        n_threads = 1
        ret = llama_eval(ctx, tokens; n_past, n_threads)
        @test ret isa Cint
    end

    @testset "token_to_str" begin
        showtestset()
        str = token_to_str(ctx, 100)
        @test str isa String
    end
end
