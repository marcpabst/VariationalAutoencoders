using Flux

abstract type AbstractVariationalAutoencoder end

function Flux.params(model::AbstractVariationalAutoencoder)
    return Flux.params(model.encoder.chain,
                        model.encoder.μ,
                        model.encoder.logσ,
                        model.decoder)
end

function Flux.cpu(model::AbstractVariationalAutoencoder)
    model = !model.use_gpu ? (return model) : deepcopy(model)
    model.use_gpu = false
    model.encoder = Flux.cpu(model.encoder)
    model.decoder = Flux.cpu(model.decoder)
    return model
end

function Flux.gpu(model::AbstractVariationalAutoencoder)
    model = model.use_gpu ? (return model) : deepcopy(model)
    model.use_gpu = true
    model.encoder = Flux.gpu(model.encoder)
    model.decoder = Flux.gpu(model.decoder)
    return model
end

function Base.show(io::IO, model::AbstractVariationalAutoencoder)
    println(io, "VAE with $(model.latent_dims) latent dimensions and $(sum(length, (Flux.params(model)))) parameters (on $(model.use_gpu ? "gpu" : "cpu")).")
end


"""
Decodes the stimulus from the latent codes by passing through the decoder network.
"""
function decode(model::AbstractVariationalAutoencoder, z) end

"""
Encodes the input by passing through the encoder network
and returns the latent codes.
"""
function encode(model::AbstractVariationalAutoencoder, x) end

"""
Forward pass through the encoder, sampling layer, and decoder
"""
function reconstruct(model::AbstractVariationalAutoencoder, x) end

"""
Calculate loss for `x` by encoding and reconstructing.
"""
function model_loss(model::AbstractVariationalAutoencoder, x) end