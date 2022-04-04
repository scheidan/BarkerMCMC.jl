# BarkerMCMC [![Build Status](https://github.com/scheidan/BarkerMCMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/scheidan/BarkerMCMC.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/scheidan/BarkerMCMC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/scheidan/BarkerMCMC.jl)


__ This is Work in Progress! __

A Monte Carlo Markov Chain sampler that makes use of gradient
information. Proposed by Livingstone et al. (2020)

The adaptative preconditioning is based on Andrieu and Thoms (2008), Algorithm 4 in Section 5.

### Installation

`] add https://github.com/scheidan/BarkerMCMC.jl`

### Usage

For details see doc string of `barker_mcmc` for all arguments.

```Julia
using BarkerMCMC
using Plots

# --- Target distribution
function log_p_rosebruck_2d(x; k=1/200)
    -k*(100*(x[2] - x[1]^2)^2 + (1 - x[1])^2)
end

function ∇log_p_rosebruck_2d(x; k=1/200)
    [-2*k*(200*x[1]^3 - 200*x[1]*x[2] + x[1] -1),   # d/dx[1]
     -200*k*(x[2] - x[1]^2)]                        # d/dx[2]
end

# --- Sampling
samp = barker_mcmc(log_p_rosebruck_2d, ∇log_p_rosebruck_2d,
                   [5.0, 5.0];
                   n_iter = 1000);


# --- Result

# acceptance rate
length(unique(samp[:,1])) / size(samp, 1)

plot(
    histogram2d(samp[:,1], samp[:,2], bins=50),
    scatter(samp[:,1], samp[:,2], alpha=0.2),

    plot(samp[:,1], label="chain x[1]"),
    plot(samp[:,2], label="chain x[2]")
)

# You may want to use MCMCChains for plots and diagonstics:
using MCMCChains
using StatsPlots

chain = Chains(samp, [:x1, :x2])
chains[200:end]                 # remove burn-in
plot(chains)
```


### References

Andrieu, C., Thoms, J., 2008. A tutorial on adaptive MCMC. Statistics and computing 18, 343–373.

Livingstone, S., Zanella, G., n.d. The Barker proposal: Combining robustness and efficiency in gradient-based MCMC. Journal of the Royal Statistical Society: Series B (Statistical Methodology) n/a. https://doi.org/10.1111/rssb.12482
