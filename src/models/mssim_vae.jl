using Flux

include("../misc/ssim.jl")
include("./backbones.jl")
include("./abstract_variational_autoencoder.jl")

"""
MSSIMVAE as proposed in arXiv:1511.06409. This model uses the multiscale structural-similarity score (MS-SSIM) as a loss 
function, which is better calibrated to human perceptual judgments of image quality. This results in images that are more 
realistic and closer to the target image than when using a pixel-wise loss (PL) function.
"""
mutable struct MSSIMVAE <: AbstractVariationalAutoencoder
    in_channels::Int
    latent_dims::Int
    window_size::Int
    window_sigma::Float32
    size_average::Bool
    encoder::@NamedTuple{chain::Chain, μ::Any, logσ::Any}
    decoder::Chain
    use_gpu::Bool
end

function MSSIMVAE(in_channels::Int, 
                  latent_dims::Int; 
                  hidden_dims::Vector{Int} = [32, 64, 128, 256, 512],
                  encoder_backbone_out_dim = hidden_dims[end]*4,
                  encoder_backbone = default_encoder_backbone64x64(in_channels, hidden_dims),
                  decoder_backbone = default_decoder_backbone64x64(latent_dims, in_channels, reverse(hidden_dims)),
                  window_size::Int = 11, 
                  window_sigma::Float32 = 1.5f0,
                  size_average::Bool = true, 
                  use_gpu = false)

    device = use_gpu ? gpu : cpu

    encoder = (
            chain = encoder_backbone |> device,
            μ  = Dense(encoder_backbone_out_dim, latent_dims) |> device, # μ
            logσ = Dense(encoder_backbone_out_dim, latent_dims)  |> device # variance
    )

    decoder = decoder_backbone |> device

    return MSSIMVAE(in_channels, 
                    latent_dims, 
                    window_size, 
                    window_sigma,
                    size_average, 
                    encoder, 
                    decoder,
                    use_gpu)
end


function model_loss(model::MSSIMVAE, x)

    device = model.use_gpu ? gpu : cpu; x = x |> device

    # reconstruct input
    μ, logσ, reconstruction = reconstruct(model, x)

    win = _fspecial_gauss_1d(model.window_size, model.window_sigma)
    win = device(repeat(win, [size(x,3); fill(1, length(size(x)) - 1 )]...))

    loss_recon = ssim(x, reconstruction; data_range = 1., win_size = 11, win = win)
    
    loss_KL =  .5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / size(x)[end]

    loss = loss_recon + loss_KL

    return (loss = loss, loss_recon = loss_recon, loss_KL = loss_KL,)
end


function encode(model::MSSIMVAE, x)
    device = model.use_gpu ? gpu : cpu
    result = model.encoder.chain(x |> device)
    z_μ, z_logσ = model.encoder.μ(result), model.encoder.logσ(result)

    return (μ = z_μ, logσ = z_logσ)
end

function decode(model::MSSIMVAE, z)
    model.decoder(z)
end

function reconstruct(model::MSSIMVAE, x)
    device = model.use_gpu ? gpu : cpu

    # encode input
    μ, logσ = encode(model, x)

    # sample from distribution
    z = cpu(μ) + randn(typeof(first(μ)), size(cpu(logσ))) .* exp.(0.5 .* cpu(logσ))

    # decode from z
    reconstuction = decode(model, device(z))

    return μ, logσ, reconstuction
end