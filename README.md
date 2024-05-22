# BarkerMCMC
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://scheidan.github.io/BarkerMCMC.jl/dev)
[![Build Status](https://github.com/scheidan/BarkerMCMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/scheidan/BarkerMCMC.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/scheidan/BarkerMCMC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/scheidan/BarkerMCMC.jl)



A Monte Carlo Markov Chain sampler that makes use of gradient
information. Proposed by Livingstone et al. (2021)

The adaptative preconditioning is based on Andrieu and Thoms (2008),
Algorithm 4 in Section 5. For details see Algorithm 7.2 of the supporting information
of Livingstone et al. (2021).

### Installation

`] add BarkerMCMC`

### Usage

The sampler can used in two ways: 

- defining the log density compatible to [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl), or
- providing two seperate functions for the log density and it's gradient.

See the [documentation](http://scheidan.github.io/BarkerMCMC.jl/dev) for examples of both approaches.


### Related Julia Packages

- [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl)
- [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl)
- [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl)

#### Hamiltonian Monte Carlo (gradient based)

- [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl)
- [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)

#### Adaptive MCMC (without gradient)

- [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl)
- [RobustAdaptiveMetropolisSampler.jl](https://github.com/anthofflab/RobustAdaptiveMetropolisSampler.jl)
- [KissMCMC.jl](https://github.com/mauro3/KissMCMC.jl)

### References

Andrieu, C., Thoms, J., 2008. A tutorial on adaptive MCMC. Statistics and computing 18, 343–373.

Livingstone, S., Zanella, G., 2021. The Barker proposal: Combining robustness and efficiency in gradient-based MCMC. Journal of the Royal Statistical Society: Series B (Statistical Methodology). https://doi.org/10.1111/rssb.12482
(see https://github.com/gzanella/barker for the R code used)
