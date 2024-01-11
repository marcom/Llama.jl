"""
    download_model(url::AbstractString; dir::AbstractString="models")

Downloads a model specified by `url` from the HuggingFace Hub into `dir` directory and returns the `model_path` to the downloaded file.
If the `dir` directory does not exist, it will be created.

Note: Currently allows only models in the GGUF format (expects the URL to end with `.gguf`).

See [HuggingFace Model Hub](https://huggingface.co/models) for a list of available models.

# Examples
```julia
# Download the Rocket model (~1GB)
url = "https://huggingface.co/ikawrakow/various-2bit-sota-gguf/resolve/main/rocket-3b-2.76bpw.gguf"
model = download_model(url) 
# Output: "models/rocket-3b-2.76bpw.gguf"
```
"""
function download_model(url::AbstractString; dir::AbstractString = "models")
    if !endswith(url, ".gguf")
        throw(ArgumentError("The provided URL is not in the GGUF format. File name: $(splitpath(url)[end])"))
    end
    if !(startswith(url, "https://huggingface.co/") ||
         startswith(url, "http://huggingface.co/"))
        @warn "Potential error. The provided URL seems to not be from the HuggingFace Hub. See https://huggingface.co/ to double-check the link."
    end

    model_path = joinpath(dir, splitpath(url)[end])
    mkpath(dirname(model_path))
    Downloads.download(url, model_path)
    return model_path
end
