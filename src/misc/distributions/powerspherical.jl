using Distributions
using SpecialFunctions
using Random
using StatsBase
using LinearAlgebra


struct PowerSpherical{T <: Real} <: ContinuousMultivariateDistribution
    μ::Vector{T}
    κ::T

    PowerSpherical(μ) =
        new{eltype(μ)}(normalize(μ) : μ, norm(μ))

    PowerSpherical(μ, κ; normalize_μ = true) =
        new{eltype(μ)}(normalize_μ ? normalize(μ) : μ, κ)
end

Base.length(d::PowerSpherical) = length(d.μ)
Base.eltype(d::PowerSpherical) = eltype(d.μ)

struct PowerSphericalSampler{T <: Real}
    μ::Vector{T}
    κ::T
    d::Int
    dist_b::Beta
    dist_u::HyperSphericalUniform
end

function sampler(d::PowerSpherical)
    dim = length(d)
    return PowerSphericalSampler(
        d.μ,
        d.κ,
        dim, 
        Beta((dim - 1) / 2. + d.κ, (dim - 1) / 2.), 
        HyperSphericalUniform(dim-1)
    )
end

Distributions._rand!(rng::AbstractRNG, d::PowerSpherical, x::AbstractVector) =
    _rand!(rng, sampler(d), x)

function _rand!(rng::AbstractRNG, spl::PowerSphericalSampler, x::AbstractVector)
    z = rand(rng, spl.dist_b)
    v = rand(rng, spl.dist_u)
    t = 2 * z - 1
    m = sqrt(1 - t ^ 2) * v'
    y = [t; m]
    e_1 = [1.; zeros(eltype(spl.μ), spl.d -1)]
    u = e_1 - spl.μ
    normalize!(u)
    x .= (-1) * (Matrix{eltype(spl.μ)}(I, spl.d, spl.d) .- 2*u*u') * y
end

function rsample(spl::PowerSphericalSampler)
    z = rand(spl.dist_b)
    v = rand(spl.dist_u)

    t = 2 * z - 1
    m = sqrt(1 - t ^ 2) * v'

    y = [t; m]
    e_1 = [1.; zeros(eltype(spl.μ), spl.d -1)]

    û = e_1 - spl.μ
    u = normalize(û)

    return -(Matrix{eltype(spl.μ)}(I, spl.d, spl.d) .- 2*u*u') * y
end

rsample(d::PowerSpherical) =
    rsample(sampler(d))

#_logpdf
function Distributions._logpdf(d::PowerSpherical, x::AbstractArray)
    a, b = (length(d) - 1) / 2. + d.κ, (length(d) - 1) / 2.
    return log(2) * (-a-b) + lgamma(a+b) - lgamma(a) + b * log(π) + d.κ .* log(d.μ' * x .+ 1)
end

# entropy
function StatsBase.entropy(d::PowerSpherical)
    a, b = (length(d) - 1) / 2. + d.κ, (length(d) - 1) / 2.
    logC = -( (a+b) * log(2) + lgamma(a) + b * log(pi) - lgamma(a+b))
    return -(logC + d.κ * ( log(2) + digamma(a) - digamma(a+b)))
end