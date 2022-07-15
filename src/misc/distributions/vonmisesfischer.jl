using SpecialFunctions
using NNlib
using LinearAlgebra
using Distributions
using Random

"""
Reparamerized ReparameterizedVonMisesFisher distribution.
"""

struct ReparameterizedVonMisesFisherSampler <: Sampleable{Multivariate,Continuous}
    p::Int          # the dimension
    κ::Float64
    b::Float64
    x0::Float64
    c::Float64
    v::Vector{Float64}
end

function ReparameterizedVonMisesFisherSampler(μ::Vector{Float64}, κ::Float64)
    p = length(μ)
    b = Distributions._vmf_bval(p, κ)
    x0 = (1.0 - b) / (1.0 + b)
    c = κ * x0 + (p - 1) * log1p(-abs2(x0))
    v = Distributions._vmf_householder_vec(μ)
    ReparameterizedVonMisesFisherSampler(p, κ, b, x0, c, v)
end

rsampler(d::VonMisesFisher) = ReparameterizedVonMisesFisherSampler(d.μ, d.κ)


function _sample_w_rej(s::ReparameterizedVonMisesFisherSampler, shape)
    κ = s.κ
    m = s.p

    c = sqrt((4 * (κ ^ 2)) + (m - 1) ^ 2)
    b_true = (-2 * κ + c) / (m - 1)

    # using Taylor approximation with a smooth swift from 10 < scale < 1
    # to avoid numerical errors for large scale
    b_app = (m - 1) / (4 * κ)

    ss = minimum([maximum([0, κ - 10]), 1])

    b = b_app * ss + b_true * (1 - ss)

    a = (m - 1 + 2 * κ + c) / 4
    d = (4 * a * b) / (1 + b) - (m - 1) * log(m - 1)


    _e, _w = _while_loop(s, b, a, d, shape; k=1)

    return _w

end

function first_nonzero(::ReparameterizedVonMisesFisherSampler, x, invalid_val=-1)
    mask = x .> 0

    idx = ifelse.(
        sum(mask, dims = length(mask)) .> 0,
        argmax(vec(maximum(float.(mask); dims=1))),
        invalid_val
    )
    return idx
end

function gather(source, index; dim = 1)
    out = ones(size(index))

    for c in CartesianIndices(index)
        t = [v for v in Tuple(c)]
        t[dim] = index[c]
        out[c] = source[CartesianIndex(t...)]
    end
    
    return out
end


function _while_loop(s::ReparameterizedVonMisesFisherSampler, b, a, d, shape; k=20, eps=1e-20)
        κ = s.κ
        m = s.p

        #  matrix while loop: samples a matrix of [A, k] samples, to avoid looping all together
        b, a, d = [
            reshape( fill(e, shape, fill(1, length(κ))...), :, 1)
            for e in (b, a, d)
        ]

        w, e, bool_mask = (
            zero(b),
            zero(b),
            (ones(size(b)) .== 1),
        )

        sample_shape = [size(b)[1], k]
        shape = length(κ)

         while sum(bool_mask) != 0
            con1 = (m - 1) / 2
            con2 = (m - 1) / 2
  
            e_ = rand(Beta(con1, con2), size(b)[1])

            u = rand(Uniform(0 + eps, 1 - eps), shape)

            w_ = (1 .- (1 .+ b) * e_) ./ (1 .- (1 .- b) .* e_)
            t = (2 .* a .* b) ./ (1 .- (1 .- b) .* e_)
            
            accept = ((m - 1.0) * log(t) - t + d) .> log.(u)
            #accept = ((self.__m - 1.0) * t.log() - t + d) > torch.log(u)

            accept_idx = first_nonzero(s, accept, -1)
            accept_idx = reshape(accept_idx, (size(accept_idx)...,1))

            accept_idx_clamped = round.(Int, clamp.(accept_idx, 1, Inf)) 
            
            # we use .abs(), in order to not get -1 index issues, the -1 is still used afterwards
            accept_idx_clamped_rs = reshape(accept_idx_clamped, :, 1) 


            w_ = gather(w_, accept_idx_clamped_rs, dim = 2)
            e_ = gather(e_, accept_idx_clamped_rs, dim = 2)

            reject = accept_idx .< 0
            accept = reject

            if any( (bool_mask .* accept))
                continue
            end

             w[(bool_mask .* accept) .+ 1] = w_[(bool_mask .* accept) .+ 1]
             e[(bool_mask .* accept) .+ 1] = e_[(bool_mask .* accept) .+ 1]

             bool_mask[(bool_mask .* accept) .+ 1] = reject[(bool_mask .* accept) .+ 1]
             
         end

         return reshape(e, shape), reshape(w, shape)
end

function switchdims(o, a, b)
    dims = collect(1:length(size(o)))

    _t = dims[a]
    dims[a] = dims[b]
    dims[b] = _t

    return permutedims(o, dims)

end


function sample(s::ReparameterizedVonMisesFisherSampler)

    w = _sample_w_rej(s, 1)

    # sample from standard normal
    v = rand(Normal(0,1), s.dist.m)

    v = switchdims(v, 1, length(size(v)))[2:end] 
    v = switchdims(v, 1, length(size(v)))
    v = v ./ mapslices(norm, v, dims = length(size(v)))

     w_ = sqrt.(clamp.(1 .- (w .^ 2), 1e-10, Inf))
     x = cat(w, w_ .* v, dims=length(size(v)))

     z = _householder_rotation(s, x)

     return z
end

function squeeze2(A)
    dropdims(A, dims = tuple(findall(size(A) .== 1)...))
end

function _householder_rotation(s::ReparameterizedVonMisesFisherSampler, x)
    μ = s.dist.μ
    e1 = [1.0 zeros((size(μ)[end] - 1))]
    u = squeeze2(e1) .- squeeze2(μ)
    u = u ./ ( mapslices(norm, u; dims = length(size(u))) .+ 1e-5)
    z = x .- 2 .* sum(x .* u, dims=length(size(x .* u))) .* u
    return z
end

Base.length(s::ReparameterizedVonMisesFisherSampler) = s.dist.m
#Base.length(s::ReparameterizedVonMisesFisher) = s.m

function Distributions._rand!(rng::AbstractRNG, s::ReparameterizedVonMisesFisherSampler, x::AbstractVector{T}) where T<:Real
    x .= sample(s)
end

Base.eltype(::ReparameterizedVonMisesFisherSampler) = Vector{Float64}


function besseli2(v, x)
    return exp(log(besselix(v,x)) + x)
end

# function entropy(d::ReparameterizedVonMisesFisher)
#         # option 1:
#         #println(d.m/2, "-", d.κ)
#         #println((d.m / 2) - 1, "-", d.κ)

#         output = (-d.κ * besseli2(d.m / 2, d.κ) / besseli2((d.m / 2) - 1, d.κ))
        
#         return output + _log_normalization(d)
# end

# function Distributions._logpdf(d::ReparameterizedVonMisesFisher, x)
#         return _log_unnormalized_prob(d, x) - _log_normalization(d)
# end

# function _log_unnormalized_prob(d::ReparameterizedVonMisesFisher, x)
#     _t = d.κ * (d.μ .* x)
#     return sum(_t, dims=length(size(_t)))
# end

# function _log_normalization(d::ReparameterizedVonMisesFisher)
#          return -(
#             (d.m / 2 - 1) * log(d.κ)
#             - (d.m / 2) * log(2 * π)
#             - (d.κ + log(besseli2(d.m / 2 - 1, d.κ)))
#         )
# end
