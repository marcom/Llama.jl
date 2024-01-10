"""
    download_model(url::AbstractString; dir::AbstractString="models")

Downloads a model specified by `url` from the HuggingFace Hub into `dir` directory and returns the path to the downloaded file.

Note: Currently works only for models in the GGUF format (expects the URL to end with `.gguf`).

See [HuggingFace Model Hub](https://huggingface.co/models) for a list of available models.

# Examples
```julia
# Download the Rocket model (~1GB)
url = "https://huggingface.co/ikawrakow/various-2bit-sota-gguf/resolve/main/rocket-3b-2.76bpw.gguf"
model = download_model(url) 
# Output: "models/rocket-3b-2.76bpw.gguf"
```
"""
function download_model(url::AbstractString; dir::AbstractString="models")
    @assert startswith(url, "https://huggingface.co/") || startswith(url, "http://huggingface.co/") "The provided URL is not a HuggingFace Hub model. See https://huggingface.co/"
    @assert endswith(url, ".gguf") "The provided URL is not in the GGUF format. File name: $(splitpath(url)[end])"

    model_fn = joinpath(dir, splitpath(url)[end]) # target file name
    mkpath(dirname(model_fn)) # ensure it exists
    Downloads.download(url, model_fn) # download the model
    return model_fn # return the path to the downloaded file
end