using LlamaCpp
using Documenter

DocMeta.setdocmeta!(LlamaCpp, :DocTestSetup, :(using LlamaCpp); recursive = true)

makedocs(;
    modules = [LlamaCpp],
    authors = "Marco Matthies <71844+marcom@users.noreply.github.com>",
    sitename = "LlamaCpp.jl",
    format = Documenter.HTML(;
        canonical = "https://marcom.github.io/LlamaCpp.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md"
    ]
)

deploydocs(;
    repo = "github.com/marcom/LlamaCpp.jl",
    devbranch = "main"
)
