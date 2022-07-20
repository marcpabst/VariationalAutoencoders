using Distributions
using Random
using StatsBase
using SpecialFunctions

struct HyperSphericalUniform <: ContinuousMultivariateDistribution
    m :: Int
end

struct HyperSphericalUniformSamplable <: Sampleable{Multivariate,Continuous}
    m :: Int
end

Base.length(s::HyperSphericalUniform) = s.m
Base.eltype(::HyperSphericalUniformSamplable) = Float32
Distributions.sampler(s::HyperSphericalUniform) = HyperSphericalUniformSamplable(s.m)

function Distributions._rand!(rng::AbstractRNG, ::HyperSphericalUniformSamplable, x::AbstractVector{T}) where T<:Real
    x .= randn(rng, T, length(x))
    normalize!(x)
end

Distributions._rand!(rng::AbstractRNG, d::HyperSphericalUniform, x) =
    _rand!(rng, sampler(d), x)


function StatsBase.entropy(s::HyperSphericalUniform)
    return 1/2 * s.m * log(pi) - lgamma(s.m/2) + log(2)
end

function Distributions._logpdf(s::HyperSphericalUniform, x)
    return -entropy(s)
end
