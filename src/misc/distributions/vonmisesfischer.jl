using Distributions
using Random
using Distributions: log2π
import Distributions._rand!
# von Mises-Fisher distribution is useful for directional statistics
#
# The implementation here follows:
#
#   - Wikipedia:
#     http://en.wikipedia.org/wiki/Von_Mises–Fisher_distribution
#
#   - R's movMF package's document:
#     http://cran.r-project.org/web/packages/movMF/vignettes/movMF.pdf
#
#   - Wenzel Jakob's notes:
#     http://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
#

logbesseli(a,b) = b + log(besselix(a, b))
besseli2(v, x) = exp(log(besselix(v,x)) + x)

struct VonMisesFisher2{T<:Real} <: ContinuousMultivariateDistribution
    μ::Vector{T}
    κ::T
    logCκ::T

    function VonMisesFisher2{T}(μ::Vector{T}, κ::T; checknorm::Bool=true) where T
        if checknorm
            Distributions.isunitvec(μ) || error("μ must be a unit vector")
        end
        κ > 0 || error("κ must be positive.")
        logCκ = vmflck(length(μ), κ)
        S = promote_type(T, typeof(logCκ))
        new{T}(Vector{S}(μ), S(κ), S(logCκ))
    end
end

VonMisesFisher2(μ::Vector{T}, κ::T) where {T<:Real} = VonMisesFisher2{T}(μ, κ)
function VonMisesFisher2(μ::Vector{T}, κ::Real) where {T<:Real}
    R = promote_type(T, eltype(κ))
    return VonMisesFisher2(convert(AbstractArray{R}, μ), convert(R, κ))
end

function VonMisesFisher2(θ::Vector)
    κ = norm(θ)
    return VonMisesFisher2(θ * (1 / κ), κ)
end

show(io::IO, d::VonMisesFisher2) = show(io, d, (:μ, :κ))

### Conversions
convert(::Type{VonMisesFisher2{T}}, d::VonMisesFisher2) where {T<:Real} = VonMisesFisher2{T}(convert(Vector{T}, d.μ), T(d.κ); checknorm=false)
Base.convert(::Type{VonMisesFisher2{T}}, d::VonMisesFisher2{T}) where {T<:Real} = d
convert(::Type{VonMisesFisher2{T}}, μ::Vector, κ, logCκ) where {T<:Real} =  VonMisesFisher2{T}(convert(Vector{T}, μ), T(κ))



### Basic properties

Base.length(d::VonMisesFisher2) = length(d.μ)

meandir(d::VonMisesFisher2) = d.μ
concentration(d::VonMisesFisher2) = d.κ

insupport(d::VonMisesFisher2, x::AbstractVector{T}) where {T<:Real} = isunitvec(x)
params(d::VonMisesFisher2) = (d.μ, d.κ)
@inline partype(d::VonMisesFisher2{T}) where {T<:Real} = T

### Evaluation

function _vmflck(p, κ)
    T = typeof(κ)
    hp = T(p/2)
    q = hp - 1
    q * log(κ) - hp * log2π - logbesseli(q, κ)
end
_vmflck3(κ) = log(κ) - log2π - κ - log1mexp(-2κ)
vmflck(p, κ) = (p == 3 ? _vmflck3(κ) : _vmflck(p, κ))

_logpdf(d::VonMisesFisher2, x::AbstractVector{T}) where {T<:Real} = d.logCκ + d.κ * dot(d.μ, x)


### Sampling

Distributions.sampler(d::VonMisesFisher2) = VonMisesFisher2Sampler(d.μ, d.κ)

function _rand!(rng::AbstractRNG, d::VonMisesFisher2, x::AbstractVector) 
    _rand!(rng, sampler(d), x)
end

function _rand!(rng::AbstractRNG, d::VonMisesFisher2, x::AbstractMatrix)
    _rand!(rng, sampler(d), x)
end


### Estimation

function fit_mle(::Type{<:VonMisesFisher2}, X::Matrix{Float64})
    r = vec(sum(X, dims=2))
    n = size(X, 2)
    r_nrm = norm(r)
    μ = rmul!(r, 1.0 / r_nrm)
    ρ = r_nrm / n
    κ = _vmf_estkappa(length(μ), ρ)
    VonMisesFisher2(μ, κ)
end

fit_mle(::Type{<:VonMisesFisher2}, X::Matrix{T}) where {T<:Real} = fit_mle(VonMisesFisher2, Float64(X))

function _vmf_estkappa(p::Int, ρ::Float64)
    # Using the fixed-point iteration algorithm in the following paper:
    #
    #   Akihiro Tanabe, Kenji Fukumizu, and Shigeyuki Oba, Takashi Takenouchi, and Shin Ishii
    #   Parameter estimation for von Mises-Fisher distributions.
    #   Computational Statistics, 2007, Vol. 22:145-157.
    #

    maxiter = 200
    half_p = 0.5 * p

    ρ2 = abs2(ρ)
    κ = ρ * (p - ρ2) / (1 - ρ2)
    i = 0
    while i < maxiter
        i += 1
        κ_prev = κ
        a = (ρ / _vmfA(half_p, κ))
        # println("i = $i, a = $a, abs(a - 1) = $(abs(a - 1))")
        κ *= a
        if abs(a - 1.0) < 1.0e-12
            break
        end
    end
    return κ
end

_vmfA(half_p::Float64, κ::Float64) = besseli(half_p, κ) / besseli(half_p - 1.0, κ)



# Sampler for von Mises-Fisher
struct VonMisesFisher2Sampler <: Sampleable{Multivariate,Continuous}
    p::Int          # the dimension
    κ::Float64
    b::Float64
    x0::Float64
    c::Float64
    v::Vector{Float64}
end

function VonMisesFisher2Sampler(μ::Vector{Float64}, κ::Float64)
    p = length(μ)
    b = _vmf_bval(p, κ)
    x0 = (1.0 - b) / (1.0 + b)
    c = κ * x0 + (p - 1) * log1p(-abs2(x0))
    v = _vmf_householder_vec(μ)
    VonMisesFisher2Sampler(p, κ, b, x0, c, v)
end

Base.length(s::VonMisesFisher2Sampler) = length(s.v)

@inline function _vmf_rot!(v::AbstractVector, x::AbstractVector)
    # rotate
    scale = 2.0 * (v' * x)
    @. x -= (scale * v)
    return x
end

@inline function _rvmf_rot!(v::AbstractVector, x::AbstractVector)
    # rotate
    scale = 2.0 * (v' * x)
    #@. x -= (scale * v)
    x = x .- (scale * v)
    return x
end

function rsample(rn,rnn, spl::VonMisesFisher2Sampler)
    w = _rvmf_genw(rn, spl)
 
    p = spl.p
    x1 = w
    s = 0.0

    x = [x1; rnn[2:p]]

    s = sum(abs2.(x[2:p]))
    # @inbounds for i = 2:p
    #     x[i] = xi = rnn[i]
    #     s += abs2(xi)
    # end

    # normalize x[2:p]
    r = sqrt((1.0 - abs2(w)) / s)
    @inbounds for i = 2:p
        x[i] *= r
    end

    x = [x[1]; x[2:p] .* r]
    # @inbounds for i = 2:p
    #     x[i] *= r
    # end

    return _rvmf_rot!(spl.v, x)
end

function _rand!(rng::AbstractRNG, spl::VonMisesFisher2Sampler, x::AbstractVector)
    
    w = _vmf_genw(rng, spl)

    p = spl.p
    x[1] = w
    s = 0.0
    @inbounds for i = 2:p
        x[i] = xi = randn(rng)
        s += abs2(xi)
    end

    # normalize x[2:p]
    r = sqrt((1.0 - abs2(w)) / s)
    @inbounds for i = 2:p
        x[i] *= r
    end

    return _vmf_rot!(spl.v, x)
end

### Core computation

_vmf_bval(p::Int, κ::Real) = (p - 1) / (2.0κ + sqrt(4 * abs2(κ) + abs2(p - 1)))

function _vmf_genw3(rng::AbstractRNG, p, b, x0, c, κ)
    ξ = rand(rng)
    w = 1.0 + (log(ξ + (1.0 - ξ)*exp(-2κ))/κ)
    return w::Float64
end

function _rvmf_genw3(rn, p, b, x0, c, κ)
    ξ = rn
    w = 1.0 + (log(ξ + (1.0 - ξ)*exp(-2κ))/κ)
    return w::Float64
end

function _vmf_genwp(rng::AbstractRNG, p, b, x0, c, κ)
    r = (p - 1) / 2.0
    betad = Beta(r, r)
    z = rand(rng, betad)
    w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
    while κ * w + (p - 1) * log(1 - x0 * w) - c < log(rand(rng))
        z = rand(rng, betad)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
    end
    return w::Float64
end

function _rvmf_genwp(rng::AbstractRNG, p, b, x0, c, κ)
    r = (p - 1) / 2.0
    betad = Beta(r, r)
    z = rand(rng, betad)
    w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
    while κ * w + (p - 1) * log(1 - x0 * w) - c < log(rand(rng))
        z = rand(rng, betad)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
    end
    return w::Float64
end

# generate the W value -- the key step in simulating vMF
#
#   following movMF's document for the p != 3 case
#   and Wenzel Jakob's document for the p == 3 case
function _vmf_genw(rng::AbstractRNG, p, b, x0, c, κ)
    if p == 3
        return _vmf_genw3(rng, p, b, x0, c, κ)
    else
        return _vmf_genwp(rng, p, b, x0, c, κ)
    end
end

_vmf_genw(rng::AbstractRNG, s::VonMisesFisher2Sampler) =
    _vmf_genw(rng, s.p, s.b, s.x0, s.c, s.κ)

function _rvmf_genw(rn, p, b, x0, c, κ)
    if p == 3
        return _rvmf_genw3(rn, p, b, x0, c, κ)
    else
        return _rvmf_genwp(rn, p, b, x0, c, κ)
    end
end


_rvmf_genw(rn, s::VonMisesFisher2Sampler) =
    _rvmf_genw(rn, s.p, s.b, s.x0, s.c, s.κ)

function _vmf_householder_vec(μ::Vector{Float64})
    # assuming μ is a unit-vector (which it should be)
    #  can compute v in a single pass over μ

    p = length(μ)
    v = similar(μ)
    v1 = μ[1] - 1.0
    s = sqrt(-2*v1)
    v1 /= s

    v = [v1; μ[2:p] ./ s]

    # @inbounds for i in 2:p
    #     v[i] = μ[i] / s
    # end

    return v
end