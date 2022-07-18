using Flux
using Distributions
using Random
using SpecialFunctions
using LinearAlgebra
using ChainRulesCore

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
    loss_recon = Flux.logitbinarycrossentropy(Flux.flatten(reconstruction), Flux.flatten(x); agg = sum)

    loss_recon = Flux.logitbinarycrossentropy(Flux.flatten(reconstruction), Flux.flatten(x), agg = identity)
    loss_recon = sum(loss_recon, dims = 1)
    loss_recon = mean(loss_recon)
    # loss_recon = sum(loss_recon; dims = 4)
    # loss_recon = mean(loss_recon)

    # KL loss
    loss_KL = mean([KL_div_stable(model.latent_dims, k) for k in vec(cpu(logκ))])

    loss = loss_recon + loss_KL

    return (loss = loss, loss_recon = loss_recon, loss_KL = loss_KL)
end

# function KL_vmf_uniform(κ, m)
#     κ * ( besseli( m / 2, κ) / besseli( (m / 2) - 1, κ) )
# end

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
    normalized_mean_dirs = normalize.(collect.(eachcol(Float64.(cpu(μ)))))
    kappas = Float64.(vec(cpu(logκ)))
    
    sample_dists = [VonMisesFisher2{Float64}(_μ, _κ, checknorm = false) for (_μ,_κ) in zip(normalized_mean_dirs, kappas)]
    
    z = Float32.(cat([rand(sd) for sd in sample_dists]..., dims = 2))

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

function KL_div(m, κ)
    return κ * besseli(m/2, κ) / besseli(m/2-1, κ) +
               log(C(m, κ)) -
               log( (2 * (pi^(m/2))) / gamma(m/2) )^(-1)

end

logbesseli(a,b) = b + log(besselix(a, b))
C(m, κ) = (κ^(m/2 - 1)) / ( (2pi)^(m/2) * besseli(m/2 - 1, κ))
logC(m, κ) = 1/2 * ( (m - 2) * log(κ) - 2 * logbesseli(m/2 - 1, κ) + m * (-log(2pi)))

"""
Numerically stable KL divergence.
"""
function KL_div_stable(m, κ)
    return κ * besselix(m / 2, κ) / besselix(m / 2 - 1, κ) +
                logC(m, κ) -
                log( 1/ ( (2 * (π^(m / 2))) / gamma(m / 2) ))
end



# ChainRulesCore.@scalar_rule(
#     KL_div_stable(m, x),
#     (
#         ChainRulesCore.@not_implemented("Not implemented...."),
#         KL_div(m, κ),
#     ),
# )

# ChainRulesCore.@scalar_rule(
#     SpecialFunctions.besselix(ν, x),
#     (
#         ChainRulesCore.@not_implemented(BESSEL_ORDER_INFO),
#         ChainRulesCore.@not_implemented(BESSEL_ORDER_INFO),
#         -(besselix(ν - 1, x) + besselix(ν + 1, x)) / 2 + Ω,
#     ),
# )

function entropy(d::VonMisesFisher2)
        m = length(d.μ)

        # C(m, κ) = (κ^(m/2 - 1)) / ( (2pi)^(m/2) * besseli(m/2-1,κ)

        # return k * besseli(m/2, k) / besseli(m/2-1, k) + log(C(m, k)) - log( (2 * (pi^(m/2))) / gamma(m/2) )^()-1)

        #return (d.κ^(m/2 - 1)) / ( (2π)^(m/2) * besseli(m/2,d.κ))

        scale = 1 / d.κ 

        output = (
            -scale 
            * besselix(m / 2, scale)
            / besselix((m / 2) - 1, scale))
        
        return output + _log_normalization(d)
end

function _log_normalization(d::VonMisesFisher2)
    scale = 1 / d.κ 
    m = length(d.μ)
    
    return -(
        (m / 2 - 1) * log(scale)
        - (m / 2) * log(2 * π)
        - (scale + log(besselix(m / 2 - 1, scale)))
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


