
using BarkerMCMC
using Test

@testset "Very basic integration test" begin
    function log_p_rosebruck_2d(x; k=1/20)
        -k*(100*(x[2] - x[1]^2)^2 + (1 - x[1])^2)
    end

    function ∇log_p_rosebruck_2d(x; k=1/20)
        [-2*k*(200*x[1]^3 - 200*x[1]*x[2] + x[1] -1),   # d/dx[1]
         -200*k*(x[2] - x[1]^2)]                        # d/dx[2]
    end

    samp = barker_mcmc(log_p_rosebruck_2d, ∇log_p_rosebruck_2d,
                       [5, -5];
                       n_iter = 100,
                       σ = 2.4/(2^(1/6)),
                       target_acceptance_rate = 0.4,
                       κ = 0.6,
                       n_iter_adaptation = Inf)

    @test size(samp) == (100, 2)

end
