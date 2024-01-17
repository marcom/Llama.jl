using Llama: download_model
@testset "download_model" begin
    # pytorch .bin model suffix
    @test_throws ArgumentError download_model("https://huggingface.co/ikawrakow/various-2bit-sota-gguf/resolve/main/rocket-3b-2.76bpw.bin")
    # not from huggingface
    log = "Potential error. The provided URL seems to not be from the HuggingFace Hub. See https://huggingface.co/ to double-check the link."
    @test_logs (:warn, log) try
        download_model("https://google.com/ikawrakow/various-2bit-sota-gguf/resolve/main/rocket-3b-2.76bpw.gguf")
    catch _
    end
    # Note: not testing the actual download because it's slow and wasteful
end
