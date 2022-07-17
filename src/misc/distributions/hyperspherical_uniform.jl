
struct HyperSphericalUniform <: ContinuousMultivariateDistribution
    m :: Int
end

struct HyperSphericalUniformSamplable <: Sampleable{Multivariate,Continuous}
    dist::HyperSphericalUniform
end

Base.length(s::HyperSphericalUniform) = s.m
Base.length(s::HyperSphericalUniformSamplable) = s.dist.m
Base.eltype(::HyperSphericalUniformSamplable) = Vector{Float64}
Distributions.sampler(s::HyperSphericalUniform) = HyperSphericalUniformSamplable(s)


function Distributions._rand!(rng::AbstractRNG, s::HyperSphericalUniformSamplable, x::AbstractVector{T}) where T<:Real
    samp = rand(Normal(0, 1), s.dist.m)
    x .= samp ./ norm(samp)
 
end

function entropy(s::HyperSphericalUniform)
    lγ = (s.m + 1) / 2
    return log(2) + ((s.m + 1) / 2) * log(π) - lγ
end

function Distributions._logpdf(s::HyperSphericalUniform, x)
    return -ones(size(x)) * entropy(s)
end
