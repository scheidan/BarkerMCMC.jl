module BarkerMCMC

export barker_mcmc

using LinearAlgebra
using ProgressMeter: @showprogress

"""
Barker MCMC

MCMC sampler that makes use of gradient information. Based on Livingstone et al. (2020)

The adaptation is based on Andrieu and Thoms (2008), Algorithm 4 in Section 5.


### Arguments

- `log_p::Function`: function returning the log of the (non-normalized) density
- `∇log_p::Function`: function returning the gradient of log of the (non-normalized) density
- `inits::Vector`: initial starting values
- `n_iter = 100`: number of iterations
- `σ = 2.4/(length(inits)^(1/6))`: global scale of proposal distribution
- `target_acceptance_rate = 0.4`: desired accept rate
- `κ = 0.6`: controls adaptation speed. Larger values lead to slower adaptation.
- `n_iter_adaptation = Inf`: number of iterations with adaptation
- `preconditioning = :eigen`: Either `:eigen` or `cholesky`. Calculating the
 preconditioning matrix with a cholesky decomposition is slighly cheaper, however, the
eigen value decomposition allows for a proper rotation of the proposal distribution.

### Return Value

An array containing the samples.


### References

Andrieu, C., Thoms, J., 2008. A tutorial on adaptive MCMC. Statistics and computing 18, 343–373.

Livingstone, S., Zanella, G., n.d. The Barker proposal: Combining robustness and efficiency in gradient-based MCMC. Journal of the Royal Statistical Society: Series B (Statistical Methodology) n/a. https://doi.org/10.1111/rssb.12482
"""
function barker_mcmc(log_p::Function, ∇log_p::Function,
                     inits::AbstractVector;
                     n_iter = 100::Int, σ = 2.4/(length(inits)^(1/6)),
                     target_acceptance_rate = 0.4, κ = 0.6,
                     n_iter_adaptation = Inf,
                     preconditioning::Symbol = :eigen)

    d = length(inits)
    chain = Array{Float64}(undef, n_iter, d)
    chain[1,:] .= inits

    log_π = log_p(inits)
    gradient = ∇log_p(inits)

    length(gradient) == d ||
        error("Size of initial values $(size(inits)) and gradient $(size(gradient)) do not match!")

    # initial value for adaptation
    log_σ = log(σ)
    μ = zeros(d)
    Σ = diagm(ones(d))

    if preconditioning == :cholesky
        M = cholesky(Σ).L
    elseif preconditioning == :eigen
        V, R = eigen(Hermitian(Σ))
        M = R * diagm(sqrt.(V))
    else
        error("Argument 'preconditioning' must be `:cholesky` or `:eigen`, not `$preconditioning`!")
    end

    @showprogress 1 "Sampling... " for t in 2:n_iter

        x = @view chain[t-1,:]

        # -- sample proposal
        xᵖ, z = barker_proposal(x, gradient, exp(log_σ), M)

        log_πᵖ = log_p(xᵖ)
        gradientᵖ =  ∇log_p(xᵖ)

        # -- acceptance probability, Alg. 7.3
        prob_accept = acceptance_prob(log_π, gradient,
                                      log_πᵖ, gradientᵖ,
                                      z, M)

        # -- accept / reject
        if rand() < prob_accept
            chain[t,:] .= xᵖ
            gradient = gradientᵖ
            log_π = log_πᵖ
        else
            chain[t,:] .= x
        end

        # -- adaptation, see Livingstone, eq(24) - eq(26)
        if t <= n_iter_adaptation
            γ = t^(-κ)              # learning rate

            log_σ += γ*(prob_accept - target_acceptance_rate)
            μ .+= γ .* (x .- μ)
            tmp = x - μ
            Σ .+= γ*(tmp * tmp' - Σ)

            if preconditioning == :cholesky
                M = cholesky(Σ).L
            else preconditioning == :eigen
                V, R = eigen(Hermitian(Σ))
                M = R * diagm(sqrt.(V))
            end

        end

    end

    return chain
end



"""
Preconditioned Barker proposal. Proposal is transformed by matrix `M` .
See Algorithm 7.2, Livingstone et al. (2020), supporting material.
"""
function barker_proposal(x::AbstractArray, gradient::AbstractArray,
                         σ::Float64, M::AbstractMatrix)

    z = σ .* randn(length(x))
    c = gradient' * M
    for i in 1:length(x)
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
    for i in 1:length(gradient)
        # same as:
        # l += log(1 + exp(-z[i] * c[i])) -
        #    log( 1 + exp( z[i] * cᵖ[i]))
        l += log1p(exp(-z[i] *  c[i])) -
            log1p(exp(  z[i] * cᵖ[i]))
    end
    return min(1, exp(l))
end


end
