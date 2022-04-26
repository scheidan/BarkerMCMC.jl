
using BarkerMCMC
using LinearAlgebra
using Statistics
using Test

@testset "Basic integration test" begin
    function log_p_rosebruck_2d(x; k=1/20)
        -k*(100*(x[2] - x[1]^2)^2 + (1 - x[1])^2)
    end

    function ∇log_p_rosebruck_2d(x; k=1/20)
        [-2*k*(200*x[1]^3 - 200*x[1]*x[2] + x[1] -1),   # d/dx[1]
         -200*k*(x[2] - x[1]^2)]                        # d/dx[2]
    end

    res = barker_mcmc(log_p_rosebruck_2d, ∇log_p_rosebruck_2d,
                      [5, -5];
                      n_iter = 100,
                      σ = 2.4/(2^(1/6)),
                      target_acceptance_rate = 0.4,
                      κ = 0.6,
                      n_iter_adaptation = Inf)

    @test size(res.samples) == (100, 2)
    @test length(res.log_p) == 100
    @test log_p_rosebruck_2d(res.samples[1,:]) ≈ res.log_p[1]
    @test log_p_rosebruck_2d(res.samples[end,:]) ≈ res.log_p[end]

end


@testset "Uncorrelated Normal" begin
    function log_p_mvnormal_ind(x)
        sd = [2^i for i in 1:length(x)]
        -sum(0.5 * (x ./ sd).^2)
    end

    function ∇log_p_mvnormal_ind(x)
        sd = [2^i for i in 1:length(x)]
        - x ./ sd.^2
    end

    # eigen value
    res = barker_mcmc(log_p_mvnormal_ind, ∇log_p_mvnormal_ind,
                      ones(5);
                      n_iter=100_000,
                      preconditioning = BarkerMCMC.precond_eigen
                      )

    means = mean(res.samples[1000:end,:], dims=1)
    for i in 1:5
        @test isapprox(0.0, means[i], atol = 2^i*0.1)
    end

    stds = std(res.samples[1000:end,:], dims=1)
    for i in 1:5
        @test isapprox.(2^i, stds[i], rtol = 0.15)
    end

    # cholesky
    res = barker_mcmc(log_p_mvnormal_ind, ∇log_p_mvnormal_ind,
                      ones(5);
                      n_iter=100_000,
                      preconditioning = BarkerMCMC.precond_cholesky
                      )

    means = mean(res.samples[1000:end,:], dims=1)
    for i in 1:5
        @test isapprox(0.0, means[i], atol =  2^i*0.1)
    end

    stds = std(res.samples[1000:end,:], dims=1)
    for i in 1:5
        @test isapprox.(2^i, stds[i], rtol = 0.15)
    end

end


@testset "Cholesky preconditioning" begin

    n = 10_000

    @testset "zero gradient, correlation" begin
        x = zeros(2)
        Σ = Hermitian([2 0.5;
                       0.5 1])
        Mpred = BarkerMCMC.precond_cholesky(Σ)
        grad = [0.0, 0.0]

        xp = hcat((
            BarkerMCMC.barker_proposal(x, grad, 1.0, Mpred)[1]
            for i in 1:n)...)

        @test mean(xp[1,:]) < 3*sqrt(2/n)
        @test mean(xp[2,:]) < 3*sqrt(1/n)
        @test norm(cov(xp') .- Σ) < 0.2
    end

    @testset "non-zero gradient, no correlation" begin
        x = zeros(2)
        Σ = Hermitian([1 0;
                       0 1])
        Mpred = BarkerMCMC.precond_cholesky(Σ)
        grad = [1000.0, 1000.0]

        xp = hcat((
            BarkerMCMC.barker_proposal(x, grad, 1.0, Mpred)[1]
            for i in 1:n)...)

        @test sum(xp[1,:] .> 0 .&& xp[2,:] .> 0 ) > 0.9*n # quadrant I

        grad = [-1000.0, 1000.0]
        xp = hcat((
            BarkerMCMC.barker_proposal(x, grad, 1.0, Mpred)[1]
            for i in 1:n)...)

        @test sum(xp[1,:] .< 0 .&& xp[2,:] .> 0 ) > 0.9*n # quadrant II

    end

end


@testset "Eigen value decomposition preconditioning" begin

    n = 10_000

    @testset "zero gradient, correlation" begin
        x = zeros(2)
        Σ = Hermitian([2 0.5;
                       0.5 1])
        Mpred = BarkerMCMC.precond_eigen(Σ)
        grad = [0.0, 0.0]

        xp = hcat((
            BarkerMCMC.barker_proposal(x, grad, 1.0, Mpred)[1]
            for i in 1:n)...)

        @test mean(xp[1,:]) < 3*sqrt(2/n)
        @test mean(xp[2,:]) < 3*sqrt(1/n)
        @test norm(cov(xp') .- Σ) < 0.2
    end

    @testset "non-zero gradient, no correlation" begin
        x = zeros(2)
        Σ = Hermitian([1 0;
                       0 1])
        Mpred = BarkerMCMC.precond_eigen(Σ)
        grad = [1000.0, 1000.0]

        xp = hcat((
            BarkerMCMC.barker_proposal(x, grad, 1.0, Mpred)[1]
            for i in 1:n)...)

        @test sum(xp[1,:] .> 0 .&& xp[2,:] .> 0 ) > 0.9*n # quadrant I

        grad = [-1000.0, 1000.0]
        xp = hcat((
            BarkerMCMC.barker_proposal(x, grad, 1.0, Mpred)[1]
            for i in 1:n)...)

        @test sum(xp[1,:] .< 0 .&& xp[2,:] .> 0 ) > 0.9*n # quadrant II

    end

end
