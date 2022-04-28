using BarkerMCMC
using Documenter

DocMeta.setdocmeta!(BarkerMCMC, :DocTestSetup, :(using BarkerMCMC); recursive=true)

makedocs(;
    modules=[BarkerMCMC],
    authors="Andreas Scheidegger <andreas.scheidegger@eawag.ch> and contributors",
    repo="https://github.com/scheidan/BarkerMCMC.jl/blob/{commit}{path}#{line}",
    sitename="BarkerMCMC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://scheidan.github.io/BarkerMCMC.jl",
        assets=String[],
    ),
    pages=[
        "BarkerMCMC" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/scheidan/BarkerMCMC.jl",
    devbranch="main",
)
