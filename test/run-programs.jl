@testset verbose=true "Llama, no model needed" begin
    showtestset()

    default_run_kwargs = Dict(:n_gpu_layers => 1, :ctx_size => 8)
    @testset "run_llama" begin
        showtestset()
        redirect_stdio(stdout=devnull) do
            @test run_llama(; model="", args=`-h`, default_run_kwargs...) isa String
            @test run_llama(; model="", prompt="", args=`-h`, default_run_kwargs...) isa String
        end
    end

    @testset "run_chat" begin
        showtestset()
        redirect_stdio(stdout=devnull) do
            @test run_chat(; model="", args=`-h`, default_run_kwargs...) isa Base.Process
        end
    end

    # @testset "LlamaContext" begin
    #     showtestset()
    #     model_path = "thisfiledoesnotexist.bin"
    #     @test_throws ErrorException LlamaContext(model_path)
    # end
end

if haskey(ENV, "LLAMA_JL_MODEL_TESTS")
    @info "ENV[\"LLAMA_JL_MODEL_TESTS\"] exists, running tests with model"
    include("run-programs-with-model.jl")
else
    @info "ENV[\"LLAMA_JL_MODEL_TESTS\"] doesn't exist, _not_ running tests with model"
end
