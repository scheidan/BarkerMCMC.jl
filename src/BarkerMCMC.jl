module BarkerMCMC

export barker_mcmc

using LinearAlgebra
import ProgressMeter
import LogDensityProblems

include("logdensityproblem_interface.jl")


"""
Adaptive MCMC sampler that makes use of gradient information. Based on Livingstone et al. (2020)
The adaptation is based on Andrieu and Thoms (2008), Algorithm 4 in Section 5.

```Julia
barker_mcmc(lp,
            inits::AbstractVector;
            n_iter = 100::Int,
            σ = 2.4/(length(inits)^(1/6)),
            proposal_scale = ones(length(inits)),
            target_acceptance_rate = 0.4,
            κ::Float64 = 0.6,
            n_iter_adaptation = Inf,
            show_progress = true,
            preconditioning::Function = BarkerMCMC.precond_eigen)
```

or
```Julia
barker_mcmc(log_p::Function, ∇log_p::Function,
            inits::AbstractVector;
            n_iter = 100::Int,
            σ = 2.4/(length(inits)^(1/6)),
            proposal_scale = ones(length(inits)),
            target_acceptance_rate = 0.4,
            κ::Float64 = 0.6,
            n_iter_adaptation = Inf,
            show_progress = true,
            preconditioning::Function = BarkerMCMC.precond_eigen)
```

### Arguments

- `lp`: log density object that supports the API of the `LogDensityProblems.jl` package
- `log_p::Function`: function returning the log of the (non-normalized) density
- `∇log_p::Function`: function returning the gradient of log of the (non-normalized) density
- `inits::Vector`: initial starting values
- `n_iter = 100`: number of iterations
- `σ = 2.4/(length(inits)^(1/6))`: global scale of proposal distribution.
- `proposal_scale = ones(length(inits))`: initial per-dimension proposal scale,
  before multiplication by the global scale `σ`: so the final scale is `σ * proposal_scale[i]`.
- `target_acceptance_rate = 0.4`: desired acceptance rate
- `κ = 0.6`: controls adaptation speed, κ ∈ (0.5, 1). Larger values lead to slower adaptation, see Section 6.1
             in Livingstone et al. (2020).
- `n_iter_adaptation = Inf`: number of iterations with adaptation
- `show_progress = true`: show progress bar?
- `preconditioning::Function = BarkerMCMC.precond_eigen`: Either `BarkerMCMC.precond_eigen` or `BarkerMCMC.precond_cholesky`. Calculating the preconditioning matrix with a cholesky decomposition is slighly cheaper, however, the eigen value decomposition allows for a proper rotation of the proposal distribution.

### Return Value

A named tuple with fields:
- `samples`: array containing the samples
- `log_p`: vector containing the value of `log_p` for each sample.

### References

Andrieu, C., Thoms, J., 2008. A tutorial on adaptive MCMC. Statistics and computing 18, 343–373.

Livingstone, S., Zanella, G., 2021. The Barker proposal: Combining robustness and efficiency in gradient-based MCMC. Journal of the Royal Statistical Society: Series B (Statistical Methodology). https://doi.org/10.1111/rssb.12482
"""
function barker_mcmc(lp,
                     inits::AbstractVector;
                     n_iter = 100::Int, σ = 2.4/(length(inits)^(1/6)),
                     proposal_scale = ones(length(inits)),
                     target_acceptance_rate = 0.4, κ::Float64 = 0.6,
                     n_iter_adaptation = Inf,
                     show_progress = true,
                     preconditioning::Function = precond_eigen)

    d = LogDensityProblems.dimension(lp)

    LogDensityProblems.capabilities(lp) >= LogDensityProblems.LogDensityOrder(1) ||
        error("The LogDensityProblem must provide gradient computation!")
    length(inits) == d ||
        error("The initial values must be of length $(d)!")
    (0.5 < κ < 1) ||
        error("κ must be in (0.5, 1)!")

    chain = Array{Float64}(undef, n_iter, d)
    log_ps = Vector{Float64}(undef, n_iter)


    chain[1,:] .= inits
    log_π, gradient = LogDensityProblems.logdensity_and_gradient(lp, inits)
    log_ps[1] = log_π

    length(gradient) == d ||
        error("Size of initial values $(size(inits)) and gradient $(size(gradient)) do not match!")

    # initial value for adaptation
    log_σ = log(σ)
    μ = zeros(d)
    Σ = diagm(proposal_scale.^2)
    tmp = similar(μ)

    M = preconditioning(Hermitian(Σ))

    p = ProgressMeter.Progress(n_iter; dt=1, desc="Sampling... ", enabled=show_progress)
    for t in 2:n_iter

        x = @view chain[t-1,:]

        # -- sample proposal
        xᵖ, z = barker_proposal(x, gradient, exp(log_σ), M)

        log_πᵖ, gradientᵖ = LogDensityProblems.logdensity_and_gradient(lp, xᵖ)

        # -- acceptance probability, Alg. 7.3
        prob_accept = acceptance_prob(log_π, gradient,
                                      log_πᵖ, gradientᵖ,
                                      z, M)

        # -- accept / reject
        if rand() < prob_accept
            chain[t,:] .= xᵖ
            gradient = gradientᵖ
            log_π = log_ps[t] = log_πᵖ
        else
            chain[t,:] .= x
            log_ps[t] = log_π
        end

        # -- adaptation, see Livingstone, eq(24) - eq(26)
        if t <= n_iter_adaptation
            γ = t^(-κ)              # learning rate
            log_σ += γ*(prob_accept - target_acceptance_rate)

            # The code below is identical to:
            # μ .+= γ .* (chain[t,:] .- μ)
            # tmp = chain[t,:] - μ
            # Σ .+= γ*(tmp * tmp' - Σ)
            @inbounds for i in eachindex(μ)
                μ[i] += γ*(chain[t,i] - μ[i])
                tmp[i] = chain[t,i] - μ[i]
            end
            @inbounds for j in axes(Σ, 2), i in axes(Σ, 1)
                Σ[i,j] += γ*(tmp[i]*tmp[j] - Σ[i,j])
            end

            M = preconditioning(Hermitian(Σ))
        end

        ProgressMeter.next!(p)
    end

    return (samples = chain, log_p = log_ps)
end

# convince wrapper so that it can be used withouth the LogDensityProblems interface
function barker_mcmc(log_p::Function, ∇log_p::Function,
                     inits::AbstractVector;
                     kwargs...)
    lp = SimpleLogDensityProblem(log_p, ∇log_p, length(inits))
    barker_mcmc(lp, inits; kwargs...)

end

"""
Preconditioned Barker proposal. Proposal is transformed by matrix `M` .
See Algorithm 7.2, Livingstone et al. (2020), supporting material.
"""
function barker_proposal(x::AbstractArray, gradient::AbstractArray,
                         σ::Float64, M::AbstractMatrix)

    z = σ .* randn(length(x))
    c = gradient' * M
    for i in eachindex(z)
        p = inv( 1 + exp(-z[i] * c[i]) )
        if rand() > p
            z[i] *= -1.0
        end
    end

    xᵖ = x + M * z
    return xᵖ, z
end


"""
Acceptance probability for preconditioned Barker proposal.
See Algorithm 7.3, Livingstone et al. (2020), supporting material.
"""
function acceptance_prob(log_π, gradient,
                         log_πᵖ, gradientᵖ,
                         z, M::AbstractMatrix)

    c =  gradient' * M
    cᵖ =  gradientᵖ' * M

    l = log_πᵖ - log_π
    for i in eachindex(gradient)
        # same as:
        # l += log(1 + exp(-z[i] * c[i])) -
        #    log( 1 + exp( z[i] * cᵖ[i]))
        l += log1pexp(-z[i] * c[i]) -
            log1pexp(  z[i] * cᵖ[i])
    end
    return min(1, exp(l))
end


"""
Given a covariance matrix Σ, computes the preconditioning matrix `M` based on cholesky decomposition.

For `M` holds that
cov(M * z) == Σ,
where `z` a uncorrelated vector of random variables with zero mean.
"""
precond_cholesky(Σ::Hermitian) = cholesky(Σ).L

"""
Given a covariance matrix Σ, computes the preconditioning matrix `M` based on eigen value decomposition.

For `M` holds that
cov(M * z) == Σ,
where `z` a uncorrelated vector of random variables with zero mean.
"""
function precond_eigen(Σ::Hermitian)
    V, R = eigen(Σ)
    λmin = eps(eltype(V)) * max(1, maximum(abs, V))
    R * Diagonal(sqrt.(max.(V, λmin)))
end


"""
Numerically stable computation of:
  `log(1+exp(x))`
"""
log1pexp(x) = max(x, 0) + log1p(exp(-abs(x)))


end
