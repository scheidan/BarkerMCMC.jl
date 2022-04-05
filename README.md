# BarkerMCMC [![Build Status](https://github.com/scheidan/BarkerMCMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/scheidan/BarkerMCMC.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/scheidan/BarkerMCMC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/scheidan/BarkerMCMC.jl)


__ This is Work in Progress! __

A Monte Carlo Markov Chain sampler that makes use of gradient
information. Proposed by Livingstone et al. (2021)

The adaptative preconditioning is based on Andrieu and Thoms (2008),
Algorithm 4 in Section 5. Algorithm 7.2 of the supporting information
of Livingstone et al. (2021) is implemented.

### Installation

`] add https://github.com/scheidan/BarkerMCMC.jl`

### Usage

See the doc string of `barker_mcmc` for all arguments.

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
samp = barker_mcmc(log_p_rosebruck_2d, ∇log_p_rosebruck_2d,
                   [5.0, 5.0];
                   n_iter = 1000);


# --- Result

# acceptance rate
length(unique(samp[:,1])) / size(samp, 1)

# You may want to use `MCMCChains.jl` for plots and diagonstics
# (must be installed separately)

using MCMCChains
using StatsPlots

chain = Chains(samp, [:x1, :x2])
chains[200:10:end]                 # remove burn-in and thinning

plot(chains)
meanplot(chain)
histogram(chain)
autocorplot(chain)
corner(chain)
```


### References

Andrieu, C., Thoms, J., 2008. A tutorial on adaptive MCMC. Statistics and computing 18, 343–373.

Livingstone, S., Zanella, G., 2021. The Barker proposal: Combining robustness and efficiency in gradient-based MCMC. Journal of the Royal Statistical Society: Series B (Statistical Methodology). https://doi.org/10.1111/rssb.12482
