
import LogDensityProblems


# Define a simple LogDensityProblem for the case were we have the
# log density and gradient directly both as functions
struct SimpleLogDensityProblem
    log_p::Function
    ∇log_p::Function
    dim::Int
end

LogDensityProblems.dimension(lp::SimpleLogDensityProblem) = lp.dim

function LogDensityProblems.capabilities(::Type{SimpleLogDensityProblem})
    LogDensityProblems.LogDensityOrder{1}()
end

LogDensityProblems.logdensity(lp::SimpleLogDensityProblem, x) = lp.log_p(x)

function LogDensityProblems.logdensity_and_gradient(lp::SimpleLogDensityProblem, x)
    (lp.log_p(x), lp.∇log_p(x))
end
