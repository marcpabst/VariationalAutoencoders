using Flux
using Distributions
using Random
using SpecialFunctions
using LinearAlgebra

"""
Hyperspherical Variational Autoencoder as proposed by 
"""
struct HypersphericalVAE <: AbstractVariationalAutoencoder
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
        μ  = Dense(encoder_backbone_out_dim, latent_dims) |> device, # μ
        logκ = Dense(encoder_backbone_out_dim, 1)  |> device, # 1/variance
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
    loss_recon = Flux.mse(x, reconstruction)

    # KL loss
    #loss_KL =  .5f0 * sum(@. (exp(2f0 * logκ) + μ^2 -1f0 - 2f0 * logκ)) / size(x)[end]

    sample_dists = VonMisesFisher2{Float64}.(
        normalize.(collect.(eachcol(Float64.(cpu(μ))))), 
        Float64.(vec(exp.(0.5 .* cpu(logκ))));
        checknorm = false
        )

    loss_KL = mean(KL.(sample_dists, [HyperSphericalUniform(length(logκ))]))

    loss = loss_recon + loss_KL

    return (loss = loss, loss_recon = loss_recon, loss_KL = loss_KL,)
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
    sample_dists = VonMisesFisher2{Float64}.(
        normalize.(collect.(eachcol(Float64.(cpu(μ))))), 
        Float64.(vec(exp.(0.5 .* cpu(logκ))));
        checknorm = false
    )
        
<<<<<<< HEAD
    z = cat(rand.(sample_dists)..., dims = 2)
=======
    z = rand.(sample_dists)
    # stack this
>>>>>>> e16ad2c505218ebe03d6f04cacc9970133c69508

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

function entropy(d::VonMisesFisher2)
        m = length(d.μ)
        output = (
            -d.κ 
            * besselix(m / 2, d.κ)
            / besselix((m / 2) - 1, d.κ))
        
        return output + _log_normalization(d)
end

function _log_normalization(d::VonMisesFisher2)
    m = length(d.μ)
    return -(
        (m / 2 - 1) * log(d.κ)
        - (m / 2) * log(2 * π)
        - (d.κ + log(besselix(m / 2 - 1, d.κ)))
    )
end

function besseli2(v, x)
    return exp(log(besselix(v,x)) + x)
end


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
    lγ = loggamma((s.m + 1) / 2)
    return log(2) + ((s.m + 1) / 2) * log(π) - lγ
end

function Distributions._logpdf(s::HyperSphericalUniform, x)
    return -ones(size(x)) * entropy(s)
end

function KL(vmf::VonMisesFisher2, hyu::HyperSphericalUniform)
    return -entropy(vmf) + entropy(hyu)
end



# import Pkg; Pkg.activate(".")
# using Distributions, VariationalAutoencoders
# vmf = VonMisesFisher([0.,0.], 1.)
# hu = HyperSphericalUniform(2)
# VariationalAutoencoders.entropy(hu)
# KL(vmf, hu)

# from hyperspherical_vae.distributions import VonMisesFisher
# from hyperspherical_vae.distributions import HypersphericalUniform
# import numpy as np
# import torch

# vmf = VonMisesFisher(torch.FloatTensor([[0., 0.]]), torch.FloatTensor([[1.]]))
# hu = HypersphericalUniform(2)


