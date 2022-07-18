using Flux
using Distributions
using Random
using SpecialFunctions
using LinearAlgebra
using ChainRulesCore
using Distributions: log1mexp

"""
Hyperspherical Variational Autoencoder as proposed by 
"""
mutable struct HypersphericalVAE <: AbstractVariationalAutoencoder
    in_channels::Int
    latent_dims::Int
    encoder::@NamedTuple{chain::Chain, μ::Any, logκ::Any}
    decoder::Chain
    use_gpu::Bool
end

function HypersphericalVAE(
                  in_channels::Int, 
                  latent_dims::Int; 
                  encoder_backbone_out_dim = 512*4,
                  encoder_backbone = default_encoder_backbone64x64(in_channels, [32, 64, 128, 256, 512]),
                  decoder_backbone = default_decoder_backbone64x64(latent_dims, in_channels, reverse([32, 64, 128, 256, 512])),
                  use_gpu = false)

    device = use_gpu ? gpu : cpu

    encoder = (
        chain = encoder_backbone |> device,
        μ  = Dense(encoder_backbone_out_dim, latent_dims, ) |> device, # μ
        logκ = Chain(Dense(encoder_backbone_out_dim, 1, softplus), x -> x .+ 1) |> device,
    )

    decoder = decoder_backbone |> device
    
    return HypersphericalVAE(in_channels, 
                    latent_dims, 
                    encoder, 
                    decoder,
                    use_gpu)
end


function model_loss(model::HypersphericalVAE, x)

    device = model.use_gpu ? gpu : cpu; x = x |> device

    μ, logκ, reconstruction = reconstruct(model, x)

    # reconstruction loss
    #loss_recon = Flux.logitbinarycrossentropy(Flux.flatten(reconstruction), Flux.flatten(x); agg = sum)

    loss_recon = Flux.logitbinarycrossentropy(Flux.flatten(reconstruction), Flux.flatten(x), agg = identity)
    loss_recon = sum(loss_recon, dims = 1)
    loss_recon = mean(loss_recon)

    # KL loss
    normalized_mean_dirs = cpu(normalize.(collect.(eachcol(μ))))
    kappas = cpu(vec(logκ))

    prior = HyperSphericalUniform(length(μ))
    dists = PowerSpherical.(normalized_mean_dirs, kappas)
    loss_KL = KL.(dists, [prior]) |> mean

    #loss_KL = mean([KL_div_stable(model.latent_dims, k) for k in vec(cpu(logκ))])

    loss = loss_recon + loss_KL

    return (loss = loss, loss_recon = loss_recon, loss_KL = loss_KL)
end

function encode(model::HypersphericalVAE, x)
    device = model.use_gpu ? gpu : cpu
    result = model.encoder.chain(x |> device)
    
    z_μ, z_logκ = model.encoder.μ(result), model.encoder.logκ(result)

    return (μ = z_μ, logκ = z_logκ)
end


function decode(model::HypersphericalVAE, z)
    model.decoder(z)
end

function reconstruct(model::HypersphericalVAE, x)
    device = model.use_gpu ? gpu : cpu

    # encode input
    μ, logκ = encode(model, x)

    # sample from distribution
    normalized_mean_dirs = cpu(normalize.(collect.(eachcol(μ))))
    kappas = cpu(vec(logκ))

    #prior = HyperSphericalUniform(length(μ))
    dists = PowerSpherical.(normalized_mean_dirs, kappas)
    z = cat(sample.(dists)..., dims = 2)

    # decode from z
    reconstuction = decode(model, device(z))

    return μ, logκ, reconstuction
end

function Flux.params(model::HypersphericalVAE)
    return Flux.params(model.encoder.chain,
                        model.encoder.μ,
                        model.encoder.logκ,
                        model.decoder)
end

# function KL_div(m, κ)
#     return κ * besseli(m/2, κ) / besseli(m/2-1, κ) +
#                log(C(m, κ)) -
#                log( (2 * (pi^(m/2))) / gamma(m/2) )^(-1)

# end

# logbesseli(a,b) = b + log(besselix(a, b))
# C(m, κ) = (κ^(m/2 - 1)) / ( (2pi)^(m/2) * besseli(m/2 - 1, κ))
# logC(m, κ) = 1/2 * ( (m - 2) * log(κ) - 2 * logbesseli(m/2 - 1, κ) + m * (-log(2pi)))

# """
# Numerically stable KL divergence.
# """
# function KL_div_stable(m, κ)
#     return κ * besselix(m / 2, κ) / besselix(m / 2 - 1, κ) +
#                 logC(m, κ) -
#                 log( 1/ ( (2 * (π^(m / 2))) / gamma(m / 2) ))
# end


# Distributions


struct HyperSphericalUniform <: ContinuousMultivariateDistribution
    m :: Int
end

struct HyperSphericalUniformSamplable <: Sampleable{Multivariate,Continuous}
    dist::HyperSphericalUniform
end

Base.length(s::HyperSphericalUniform) = s.m
Base.length(s::HyperSphericalUniformSamplable) = s.dist.m
Base.eltype(::HyperSphericalUniformSamplable) = Vector{Float32}
Distributions.sampler(s::HyperSphericalUniform) = HyperSphericalUniformSamplable(s)

function Distributions._rand!(rng::AbstractRNG, s::HyperSphericalUniformSamplable, x::AbstractVector{T}) where T<:Real
    samp = rand(Normal(0, 1), s.dist.m)
end

function entropy(s::HyperSphericalUniform)
    return -(lgamma( s.m / 2) - log(2.) + (s.m / 2) * log(pi))
end

function Distributions._logpdf(s::HyperSphericalUniform, x)
    return -ones(size(x)) * entropy(s)
end

struct PowerSpherical{T<:Real}
    μ::Vector{T}
    κ::T
    d::Int
    dist1::Beta
    dist2::HyperSphericalUniform
end

function PowerSpherical(μ::Vector{T}, κ::T) where T <: Real
    d = length(μ)
    dist1 = Beta((d - 1) / 2, (d - 1) / 2 + κ)
    dist2 = HyperSphericalUniform(d-1)
    return PowerSpherical(μ, κ, d, dist1, dist2)
end

function sample(d::PowerSpherical)
    z = rand(d.dist1)
    v = rand(d.dist2)

    μ = d.μ
    t = 2 * z - 1

    m = sqrt(1 - t ^ 2) * v'

    y = [t; m]
    e_1 = [1.; zeros(d.d-1)]

    û = e_1 - μ

    u = normalize(û)
    
    x = (Matrix(I, d.d, d.d) .- 2*u*u') * y

    normalize(x)
end

function log_normalizer(d::PowerSpherical)
    alpha, beta = params(d.dist1)
    return -((alpha + beta) * log(2) + lgamma(alpha) - lgamma(alpha + beta) + beta * log(pi))
end

function entropy(d::PowerSpherical)
    alpha, beta = params(d.dist1)
    return (log_normalizer(d) + d.κ * (log(2) + digamma(alpha) - digamma(alpha + beta)))
end

function KL(p::PowerSpherical, q::HyperSphericalUniform)
    return -entropy(p) + entropy(q)
end