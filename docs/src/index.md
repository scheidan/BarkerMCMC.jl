```@meta
CurrentModule = BarkerMCMC
```

# BarkerMCMC.jl


Implements an adaptive Monte Carlo Markov Chain sampler that makes use
of gradient information. It was the proposed by Livingstone et
al. (2021).

The adaptative preconditioning is based on Andrieu and Thoms (2008),
Algorithm 4 in Section 5. We followed the Algorithm 7.2 of the
supporting information of Livingstone et al. (2021) with slight
modifications.


You can find the repository with the source code [here](https://github.com/scheidan/BarkerMCMC.jl).


## Usage

You can either define the log density compatible to
[LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl),
or you provide the log density and it's
gradient as two separate functions.

### `LogDensityProblems` interface

This approach is recommend to for most modeling tasks. This example
demonstrate Bayesian inference of a simple regression. Note, that we
use `TransformVariables.jl` to ensure that the
standard deviation is always positive. The package takes care of the
determinate correction.

```
using BarkerMCMC

using TransformVariables: transform, inverse, as, asℝ, asℝ₊
using TransformedLogDensities: TransformedLogDensity
using LogDensityProblemsAD: ADgradient
using Distributions

# -----------
# simulate data

x = rand(10)
y = 2 .+ 1.5*x .+ randn(10)*0.2


# -----------
# define model

model(x, θ) = θ.a + θ.b*x

function likelihood(θ, x, y)
    ll = 0.0
    for i in eachindex(y)
        ll += logpdf(Normal(model(x[i], θ), θ.σ), y[i])
    end
    ll
end

function prior(θ)
    logpdf(Normal(0,1), θ.a) +
        logpdf(Normal(0,1), θ.b) +
        logpdf(Exponential(1), θ.σ)
end

posterior(θ, x, y) = likelihood(θ, x, y) + prior(θ)

# transformation σ to [0, ∞)
trans = as((a = asℝ, b = asℝ, σ = asℝ₊))

# -----------
# define lp

lp = TransformedLogDensity(trans, θ -> posterior(θ, x, y))

# define gradient with AD
lp = ADgradient(:ForwardDiff, lp)
# lp= ADgradient(:Zygote, lp)  # we can use different AD backends


# -----------
# sample

# we need the inits in transformed space
inits = inverse(trans, (a=0, b=2, σ=0.5))

results = barker_mcmc(lp,
                      inits;
                      n_iter = 10_000)

# back-transform samples to original parameter space
samples = [transform(trans, s)
           for s in eachrow(results.samples)]

# convert to array
samplesArray = vcat((hcat(i...) for i in samples)...)
```

See the example below how the results can be visualized.


### Function interface
When not parameter transformations are required, the function interface
can be a bit simpler to work with. Here we sample from the 'banana-shaped' Rosenbruck function:

```Julia
using BarkerMCMC

# --- Define target distribution and it's gradient
#     (or use automatic differentation)

function log_p_rosebruck_2d(x; k=1/200)
    -k*(100*(x[2] - x[1]^2)^2 + (1 - x[1])^2)
end

function ∇log_p_rosebruck_2d(x; k=1/200)
    [-2*k*(200*x[1]^3 - 200*x[1]*x[2] + x[1] -1),   # d/dx[1]
     -200*k*(x[2] - x[1]^2)]                        # d/dx[2]
end

# --- Generate samples

res = barker_mcmc(log_p_rosebruck_2d,
                  ∇log_p_rosebruck_2d,
                  [5.0, 5.0];
                  n_iter = 1_000,
                  target_acceptance_rate=0.4)

res.samples
res.log_p

# --- Visualize results

# acceptance rate
length(unique(res.samples[:,1])) / size(res.samples, 1)

# You may want to use `MCMCChains.jl` for plots and diagonstics
# (must be installed separately)

using MCMCChains
using StatsPlots

chain = Chains(res.samples, [:x1, :x2])
chains[200:10:end]                 # remove burn-in and apply thinning

plot(chains)
meanplot(chain)
histogram(chain)
autocorplot(chain)
corner(chain)
```

## API

```@docs
barker_mcmc
```

```@docs
precond_cholesky
precond_eigen
```

## Related Julia Packages

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


## Literature


Andrieu, C., Thoms, J., 2008. A tutorial on adaptive MCMC. Statistics and computing 18, 343–373.

Livingstone, S., Zanella, G., 2021. The Barker proposal: Combining robustness and efficiency in gradient-based MCMC. Journal of the Royal Statistical Society: Series B (Statistical Methodology). <https://doi.org/10.1111/rssb.12482>
