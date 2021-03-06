# non-mutating versions of samplers

using Distributions
using Distributions: VonMisesFisherSampler, PowerSphericalSampler
using Random

rsampler(s::Distributions.Distribution) = Distributions.sampler(s)
rrand(rng::AbstractRNG, s::Distributions.Distribution) = rrand(rng, rsampler(s))
rrand(s::Distributions.Distribution) = rrand(Random.GLOBAL_RNG, rsampler(s))

function rsampler(s::VonMisesFisher) 
    p = length(s.μ)
    b = _vmf_bval(p, s.κ)
    x0 = (1.0 - b) / (1.0 + b)
    c = s.κ * x0 + (p - 1) * log1p(-abs2(x0))
    v = _vmf_householder_vec(s.μ)

    VonMisesFisherSampler(p, s.κ, b, x0, c, v)
end

function rrand(rng::AbstractRNG, spl::PowerSphericalSampler)
    z = rand(rng, spl.dist_b)
    v = rand(rng, spl.dist_u)

    t = 2 * z - 1
    m = sqrt(1 - t ^ 2) * v'

    y = [t; m]
    e_1 = [1.; zeros(eltype(spl.μ), length(spl) - 1)]

    û = e_1 .- spl.μ

    u = normalize(û)

    return (-1) * (I(length(spl)) .- 2*u*u') * y
end


function rrand(rng::AbstractRNG, spl::VonMisesFisherSampler)
    w = _vmf_genw(rng, spl)
    p = spl.p

    x = [w; randn(rng, p-1)]
    s = sum(abs2.(x[2:p]))

    r = sqrt((1.0 - abs2(w)) / s)

    x = [x[1]; x[2:p] .* r]

    return _vmf_rot(spl.v, x)
end

### Core computation
function _vmf_rot(v::AbstractVector, x::AbstractVector)
    # rotate
    scale = 2.0 * (v' * x)

    t = scale .* v


    y = [x[i] - t[i] for i in 1:length(x)]

    #@. x -= (scale * v)
    return y
end

_vmf_bval(p::Int, κ::Real) = (p - 1) / (2.0κ + sqrt(4 * abs2(κ) + abs2(p - 1)))

function _vmf_genw3(rng::AbstractRNG, p, b, x0, c, κ)
    ξ = rand(rng)
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

_vmf_genw(rng::AbstractRNG, s::VonMisesFisherSampler) =
    _vmf_genw(rng, s.p, s.b, s.x0, s.c, s.κ)

function _vmf_householder_vec(μ::Vector{Float64})
    # assuming μ is a unit-vector (which it should be)
    #  can compute v in a single pass over μ

    p = length(μ)
    # v = similar(μ)
    # v[1] = μ[1] - 1.0
    # s = sqrt(-2*v[1])
    # v[1] /= s

    # @inbounds for i in 2:p
    #     v[i] = μ[i] / s
    # end

    v1 = μ[1] - 1.0
    s = sqrt(-2*v1)
    v1 = v1 / s
    v = [v1; μ[2:p] ./ s]
    
    return v
end
