using Flux

"""
β-VAE as proposed by Higgins et al, ICLR, 2017. The beta-VAE is a new state-of-the-art framework 
for automated discovery of interpretable factorised latent representations from raw image data in
a completely unsupervised manner. This approach is a modification of the variational autoencoder (VAE) 
framework, and introduces an adjustable hyperparameter beta that balances latent channel capacity and 
independence constraints with reconstruction accuracy.
"""
struct BetaVEA
    in_channels::Int
    latent_dims::Int
    encoder::@NamedTuple{chain::Chain, μ::Any, logσ::Any}
    decoder::Chain
    beta::Int
    use_gpu::Bool
end

function BetaVEA(in_channels::Int, 
                  latent_dims::Int; 
                  hidden_dims::Vector{Int} = [32, 64, 128, 256, 512],
                  encoder_backbone_out_dim = hidden_dims[end]*4,
                  encoder_backbone = default_encoder_backbone64x64(in_channels, hidden_dims),
                  decoder_backbone = default_decoder_backbone64x64(latent_dims, in_channels, reverse(hidden_dims)),
                  beta::Int = 4,
                  use_gpu = false)

    device = use_gpu ? gpu : cpu

    encoder = (
        chain = encoder_backbone |> device,
        μ  = Dense(encoder_backbone_out_dim, latent_dims) |> device, # μ
        logσ = Dense(encoder_backbone_out_dim, latent_dims)  |> device # variance
    )

    decoder = decoder_backbone |> device
    
    return BetaVEA(in_channels, 
                    latent_dims, 
                    hidden_dims, 
                    encoder, 
                    decoder,
                    beta,
                    use_gpu)
end


function model_loss(model::BetaVEA, x)

    device = model.use_gpu ? gpu : cpu; x = x |> device

    μ, logσ, reconstruction = reconstruct(model, x)

    # reconstruction loss
    loss_recon = Flux.mse(X, reconstruction)

    # KL loss
    loss_KL =  .5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / size(x)[end]

    loss = loss_recon + model.beta * loss_KL

    return (loss = loss, loss_recon = recons_loss, loss_KL = KL_loss,)
end


function encode(model::BetaVEA, x)
    device = model.use_gpu ? gpu : cpu
    result = model.encoder.chain(x |> device)
    
    z_μ, z_logσ = model.encoder.μ(result), model.encoder.logσ(result)

    return (μ = z_μ, logσ = z_logσ)
end


function decode(model::BetaVEA, z)
    model.decoder(z)
end

function reconstruct(model::BetaVEA, x)
    device = model.use_gpu ? gpu : cpu

    # encode input
    μ, logσ = encode(model, x)

    # sample from distribution
    z = cpu(μ) + randn(typeof(first(μ)), size(cpu(logσ))) .* exp.(0.5 .* cpu(logσ))

    # decode from z
    reconstuction = decode(model, device(z))

    return μ, logσ, reconstuction
end