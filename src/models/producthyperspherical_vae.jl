using Flux
using Distributions
using Random
using SpecialFunctions
using LinearAlgebra
using CUDA
using ChainRulesCore
using StatsBase
using Distributions: log1mexp


"""
Hyperspherical Variational Autoencoder as proposed by 
"""
mutable struct ProductHypersphericalVAE <: AbstractVariationalAutoencoder
    in_channels::Int
    latent_dims::Int
    encoder::@NamedTuple{chain::Chain, θ::Any, κ::Any}
    decoder::Chain
    use_gpu::Bool
end

function ProductHypersphericalVAE(
                  in_channels::Int, 
                  latent_dims::Int; 
                  encoder_backbone_out_dim = 512*4,
                  encoder_backbone = default_encoder_backbone64x64(in_channels, [32, 64, 128, 256, 512]),
                  decoder_backbone = default_decoder_backbone64x64(latent_dims*2, in_channels, reverse([32, 64, 128, 256, 512])),
                  use_gpu = false,
                  seed = 42)
    

    Random.seed!(seed); CUDA.seed!(seed)

    device = use_gpu ? gpu : cpu

    encoder = (
        chain = encoder_backbone |> device,
        θ  = Dense(encoder_backbone_out_dim, latent_dims) |> device, # mean angle
        κ = Chain(Dense(encoder_backbone_out_dim, latent_dims, softplus), x -> x .+ 1) |> device, # concentration
    )

    decoder = decoder_backbone |> device
    
    return ProductHypersphericalVAE(in_channels, 
                    latent_dims, 
                    encoder, 
                    decoder,
                    use_gpu)
end


function model_loss(model::ProductHypersphericalVAE, x)

    device = model.use_gpu ? gpu : cpu; x = x |> device

    θ, κ, reconstruction = reconstruct(model, x)

    # reconstruction loss

    loss_recon = Flux.logitbinarycrossentropy(Flux.flatten(reconstruction), Flux.flatten(x), agg = identity)
    loss_recon = sum(loss_recon, dims = 1)
    loss_recon = mean(loss_recon)

    # KL loss

    prior = HyperSphericalUniform(model.latent_dims)

    loss_KL = 0f0

    for d in model.latent_dims
        thetas = Float64.(cpu(vec(θ[d,:])))
        kappas = Float64.(cpu(vec(κ[d,:])))
    
        dists = [VonMisesFisher{Float64}([cos(_theta), sin(_theta)], _kappa; checknorm = false) for (_theta,_kappa) in zip(thetas, kappas)]

        loss_KL += kldivergence.(dists, [prior]) |> mean

    end

    loss = Float32(loss_recon + loss_KL)

    return (loss = loss, loss_recon = loss_recon, loss_KL = loss_KL)
end

function encode(model::ProductHypersphericalVAE, x)
    device = model.use_gpu ? gpu : cpu
    result = model.encoder.chain(x |> device)
    
    z_μ, z_logκ = model.encoder.θ(result), model.encoder.κ(result)

    return (μ = z_μ, logκ = z_logκ)
end


function decode(model::ProductHypersphericalVAE, z)
    model.decoder(z)
end

function reconstruct(model::ProductHypersphericalVAE, x)
    device = model.use_gpu ? gpu : cpu

    # encode input
    θ, κ = encode(model, x)

    z = zeros(size(θ))

    # sample from distribution
    for d in model.latent_dims
        thetas = Float64.(cpu(vec(θ[d,:])))
        kappas = Float64.(cpu(vec(κ[d,:])))
    
        dists = [VonMisesFisher{Float64}([cos(_theta), sin(_theta)], _kappa; checknorm = false) for (_theta,_kappa) in zip(thetas, kappas)]

        z[d,:] = hcat([atan(rrand(d)...) for d in dists]...)
    end

    #dists = [VonMisesFisher{Float64}(normalize(_mu), _kappa; checknorm = false) for (_mu,_kappa) in zip(mean_dirs, kappas)]
    #dists = [PowerSpherical(_mu, _kappa; check_args = false, normalize_μ = true) for (_mu,_kappa) in zip(mean_dirs, kappas)]
    #z = hcat([rrand(d) for d in dists]...) |> device

    # decode from z
    reconstuction = decode(model, device(Float32.(z)))

    return θ, κ, reconstuction
end

function Flux.params(model::ProductHypersphericalVAE)
    return Flux.params(model.encoder.chain,
                        model.encoder.θ,
                        model.encoder.κ,
                        model.decoder)
end


logbesseli(a,b) = b + log(besselix(a, b))
# C(m, κ) = (κ^(m/2 - 1)) / ( (2pi)^(m/2) * besseli(m/2 - 1, κ))
logC(m, κ) = 1/2 * ( (m - 2) * log(κ) - 2 * logbesseli(m/2 - 1, κ) + m * (-log(2pi)))

"""
Numerically stable KL divergence.
"""
function StatsBase.kldivergence(p::VonMisesFisher, q::HyperSphericalUniform)
    κ, m = p.κ, q.d

    return κ * besselix(m / 2, κ) / besselix(m / 2 - 1, κ) +
                logC(m, κ) -
                log( 1/ ( (2 * (π^(m / 2))) / gamma(m / 2) ))
end


# Distributions
function KL(p::PowerSpherical, q::HyperSphericalUniform)
    return -entropy(p) + entropy(q)
end

