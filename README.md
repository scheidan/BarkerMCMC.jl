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

See the documentation for all arguments.

```Julia
using BarkerMCMC

# --- Target distribution
function log_p_rosebruck_2d(x; k=1/200)
    -k*(100*(x[2] - x[1]^2)^2 + (1 - x[1])^2)
end

function ∇log_p_rosebruck_2d(x; k=1/200)
    [-2*k*(200*x[1]^3 - 200*x[1]*x[2] + x[1] -1),   # d/dx[1]
     -200*k*(x[2] - x[1]^2)]                        # d/dx[2]
end

# --- Sampling
# see `?barker_mcmc` for all options
res = barker_mcmc(log_p_rosebruck_2d,
                  ∇log_p_rosebruck_2d,
                  [5.0, 5.0];
                  n_iter = 1_000,
                  target_acceptance_rate=0.4)

res.samples
res.log_p

# --- Result

# acceptance rate
length(unique(res.samples[:,1])) / size(res.samples, 1)

# You may want to use `MCMCChains.jl` for plots and diagonstics
# (must be installed separately)

using MCMCChains
using StatsPlots

chain = Chains(res.samples, [:x1, :x2])
chains[200:10:end]                 # remove burn-in and thinning

plot(chains)
meanplot(chain)
histogram(chain)
autocorplot(chain)
corner(chain)
```

### Related Julia Packages

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
