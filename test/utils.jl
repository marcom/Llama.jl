using Llama: download_model
@testset "download_model" begin
    # pytorch .bin model suffix
    @test_throws AssertionError download_model("https://huggingface.co/ikawrakow/various-2bit-sota-gguf/resolve/main/rocket-3b-2.76bpw.bin")
    # not from huggingface
    @test_throws AssertionError download_model("https://google.com/ikawrakow/various-2bit-sota-gguf/resolve/main/rocket-3b-2.76bpw.gguf")
    # Note: not testing the actual download because it's slow and wasteful
end