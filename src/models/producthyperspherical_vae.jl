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
    encoder::@NamedTuple{chain::Chain, μ::Any, logκ::Any}
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
        μ  = Dense(encoder_backbone_out_dim, latent_dims*2, ) |> device, # μ
        logκ = Chain(Dense(encoder_backbone_out_dim, latent_dims, softplus), x -> x .+ 1) |> device,
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

    μ, logκ, reconstruction = reconstruct(model, x)

    # reconstruction loss
    #loss_recon = Flux.logitbinarycrossentropy(reconstruction, x; agg = sum) / size(x,4)

    loss_recon = Flux.logitbinarycrossentropy(Flux.flatten(reconstruction), Flux.flatten(x), agg = identity)
    loss_recon = sum(loss_recon, dims = 1)
    loss_recon = mean(loss_recon)

    # KL loss

    prior = HyperSphericalUniform(model.latent_dims)

    loss_KL = 0f0

    for d in model.latent_dims
        mean_dirs = cpu(collect.(eachcol(μ[d*2:d*2+1,:])))
        kappas = cpu(vec(logκ[d,:]))
    
        dists = [VonMisesFisher{Float32}(normalize(_mu), _kappa; checknorm = false) for (_mu,_kappa) in zip(mean_dirs, kappas)]

        loss_KL += kldivergence.(dists, [prior]) |> mean

    end

    loss = loss_recon + loss_KL

    return (loss = loss, loss_recon = loss_recon, loss_KL = loss_KL)
end

function encode(model::ProductHypersphericalVAE, x)
    device = model.use_gpu ? gpu : cpu
    result = model.encoder.chain(x |> device)
    
    z_μ, z_logκ = model.encoder.μ(result), model.encoder.logκ(result)

    return (μ = z_μ, logκ = z_logκ)
end


function decode(model::ProductHypersphericalVAE, z)
    model.decoder(z)
end

function reconstruct(model::ProductHypersphericalVAE, x)
    device = model.use_gpu ? gpu : cpu

    # encode input
    μ, logκ = encode(model, x)

    # sample from distribution
    mean_dirs = cpu(collect.(eachcol(Float64.(μ))))
    kappas = cpu(vec(Float64.(logκ)))

    dists = [VonMisesFisher{Float64}(normalize(_mu), _kappa; checknorm = false) for (_mu,_kappa) in zip(mean_dirs, kappas)]
    #dists = [PowerSpherical(_mu, _kappa; check_args = false, normalize_μ = true) for (_mu,_kappa) in zip(mean_dirs, kappas)]
    z = hcat([rrand(d) for d in dists]...) |> device

    # decode from z
    reconstuction = decode(model, Float32.(z))

    return μ, logκ, reconstuction
end

function Flux.params(model::ProductHypersphericalVAE)
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

