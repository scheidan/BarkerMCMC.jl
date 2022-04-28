var documenterSearchIndex = {"docs":
[{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"CurrentModule = BarkerMCMC","category":"page"},{"location":"#BarkerMCMC.jl","page":"BarkerMCMC","title":"BarkerMCMC.jl","text":"","category":"section"},{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"Implements an adaptive Monte Carlo Markov Chain sampler that makes use of gradient information. It was the proposed by Livingstone et al. (2021).","category":"page"},{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"The adaptative preconditioning is based on Andrieu and Thoms (2008), Algorithm 4 in Section 5. We followed the Algorithm 7.2 of the supporting information of Livingstone et al. (2021) with slight modifications.","category":"page"},{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"You can find the repository with the source code here.","category":"page"},{"location":"#Example","page":"BarkerMCMC","title":"Example","text":"","category":"section"},{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"Here we sample from the 'banana-shaped' Rosenbruck function:","category":"page"},{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"using BarkerMCMC\n\n# --- Define target distribution and it's gradient\n\nfunction log_p_rosebruck_2d(x; k=1/200)\n    -k*(100*(x[2] - x[1]^2)^2 + (1 - x[1])^2)\nend\n\nfunction ∇log_p_rosebruck_2d(x; k=1/200)\n    [-2*k*(200*x[1]^3 - 200*x[1]*x[2] + x[1] -1),   # d/dx[1]\n     -200*k*(x[2] - x[1]^2)]                        # d/dx[2]\nend\n\n# --- Sampling\n\nres = barker_mcmc(log_p_rosebruck_2d,\n                  ∇log_p_rosebruck_2d,\n                  [5.0, 5.0];\n                  n_iter = 1_000,\n                  target_acceptance_rate=0.4)\n\nres.samples\nres.log_p\n\n# --- Result\n\n# acceptance rate\nlength(unique(res.samples[:,1])) / size(res.samples, 1)\n\n# You may want to use `MCMCChains.jl` for plots and diagonstics\n# (must be installed separately)\n\nusing MCMCChains\nusing StatsPlots\n\nchain = Chains(res.samples, [:x1, :x2])\nchains[200:10:end]                 # remove burn-in and apply thinning\n\nplot(chains)\nmeanplot(chain)\nhistogram(chain)\nautocorplot(chain)\ncorner(chain)","category":"page"},{"location":"#API","page":"BarkerMCMC","title":"API","text":"","category":"section"},{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"barker_mcmc","category":"page"},{"location":"#BarkerMCMC.barker_mcmc","page":"BarkerMCMC","title":"BarkerMCMC.barker_mcmc","text":"Adaptive MCMC sampler that makes use of gradient information. Based on Livingstone et al. (2020) The adaptation is based on Andrieu and Thoms (2008), Algorithm 4 in Section 5.\n\nbarker_mcmc(log_p::Function, ∇log_p::Function,\n            inits::AbstractVector;\n            n_iter = 100::Int,\n            σ = 2.4/(length(inits)^(1/6)),\n            target_acceptance_rate = 0.4,\n            κ::Float64 = 0.6,\n            n_iter_adaptation = Inf,\n            preconditioning::Function = BarkerMCMC.precond_eigen)\n\nArguments\n\nlog_p::Function: function returning the log of the (non-normalized) density\n∇log_p::Function: function returning the gradient of log of the (non-normalized) density\ninits::Vector: initial starting values\nn_iter = 100: number of iterations\nσ = 2.4/(length(inits)^(1/6)): global scale of proposal distribution\ntarget_acceptance_rate = 0.4: desired accept rate\nκ = 0.6: controls adaptation speed. Larger values lead to slower adaptation, see Livingstone et al. (2020).\nn_iter_adaptation = Inf: number of iterations with adaptation\npreconditioning::Function = BarkerMCMC.precond_eigen: Either BarkerMCMC.precond_eigen or BarkerMCMC.precond_cholesky. Calculating the preconditioning matrix with a cholesky decomposition is slighly cheaper, however, the eigen value decomposition allows for a proper rotation of the proposal distribution.\n\nReturn Value\n\nA named tuple with fields:\n\nsamples: array containing the samples\nlog_p: vector containing the value of log_p for each sample.\n\nReferences\n\nAndrieu, C., Thoms, J., 2008. A tutorial on adaptive MCMC. Statistics and computing 18, 343–373.\n\nLivingstone, S., Zanella, G., 2021. The Barker proposal: Combining robustness and efficiency in gradient-based MCMC. Journal of the Royal Statistical Society: Series B (Statistical Methodology). https://doi.org/10.1111/rssb.12482\n\n\n\n\n\n","category":"function"},{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"precond_cholesky\nprecond_eigen","category":"page"},{"location":"#BarkerMCMC.precond_cholesky","page":"BarkerMCMC","title":"BarkerMCMC.precond_cholesky","text":"Given a covariance matrix Σ, computes the preconditioning matrix M based on cholesky decomposition.\n\nFor M holds that cov(M * z) == Σ, where z a uncorrelated vector of random variables with zero mean.\n\n\n\n\n\n","category":"function"},{"location":"#BarkerMCMC.precond_eigen","page":"BarkerMCMC","title":"BarkerMCMC.precond_eigen","text":"Given a covariance matrix Σ, computes the preconditioning matrix M based on eigen value decomposition.\n\nFor M holds that cov(M * z) == Σ, where z a uncorrelated vector of random variables with zero mean.\n\n\n\n\n\n","category":"function"},{"location":"#Related-Julia-Packages","page":"BarkerMCMC","title":"Related Julia Packages","text":"","category":"section"},{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"MCMCChains.jl","category":"page"},{"location":"#Hamiltonian-Monte-Carlo-(gradient-based)","page":"BarkerMCMC","title":"Hamiltonian Monte Carlo (gradient based)","text":"","category":"section"},{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"DynamicHMC.jl\nAdvancedHMC.jl","category":"page"},{"location":"#Adaptive-MCMC-(without-gradient)","page":"BarkerMCMC","title":"Adaptive MCMC (without gradient)","text":"","category":"section"},{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"AdaptiveMCMC.jl\nRobustAdaptiveMetropolisSampler.jl\nKissMCMC.jl","category":"page"},{"location":"#Literature","page":"BarkerMCMC","title":"Literature","text":"","category":"section"},{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"Andrieu, C., Thoms, J., 2008. A tutorial on adaptive MCMC. Statistics and computing 18, 343–373.","category":"page"},{"location":"","page":"BarkerMCMC","title":"BarkerMCMC","text":"Livingstone, S., Zanella, G., 2021. The Barker proposal: Combining robustness and efficiency in gradient-based MCMC. Journal of the Royal Statistical Society: Series B (Statistical Methodology). https://doi.org/10.1111/rssb.12482","category":"page"}]
}
