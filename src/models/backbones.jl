using Flux

"""
Convolutional backbone for encoding 64 by 64 images.
"""
function default_encoder_backbone64x64(in_channels::Int, hidden_dims::Vector{Int}; activation = Flux.leakyrelu)
    encoder_chain = []

    for h_dim in hidden_dims
        push!(encoder_chain,
                Conv((3,3), in_channels => h_dim; stride= 2, pad = 1),
                BatchNorm(h_dim, activation)
            )
            in_channels = h_dim
    end

    # add flatten layer
    push!(encoder_chain, Flux.flatten)

    return Flux.Chain(encoder_chain...)
end

"""
Convolutional backbone for decoding 64 by 64 images.
"""
function default_decoder_backbone64x64(latent_dims::Int, out_channels::Int, rhidden_dims::Vector{Int}; activation = Flux.leakyrelu, in_dimensions = (2,2))

    decoder_chain = [Dense(latent_dims => rhidden_dims[1] * prod(in_dimensions)),
                     x -> reshape(x, in_dimensions[1], in_dimensions[2], rhidden_dims[1], :)]

    # add decoder layers
    for i in 1:length(rhidden_dims)-1
        push!(decoder_chain,
                ConvTranspose((3,3), rhidden_dims[i] => rhidden_dims[i + 1], stride = 2, pad = SamePad()),
                BatchNorm(rhidden_dims[i + 1], activation)
        )
    end

    # add final layers
    push!(decoder_chain,
        ConvTranspose((3,3), rhidden_dims[end] => rhidden_dims[end], stride=2, pad = SamePad()),
        BatchNorm(rhidden_dims[end], activation),
        Conv((3,3), rhidden_dims[end] => out_channels, tanh, pad = 1)
    )
    
    return Flux.Chain(decoder_chain...)
end


function small_encoder_backbone64x64(in_channels::Int, hidden_dims::Vector{Int}; activation = Flux.leakyrelu)
    encoder_chain = []

    for h_dim in hidden_dims
        push!(encoder_chain,
                Conv((4,4), in_channels => h_dim, activation; stride= 2, pad = 1)
            )
            in_channels = h_dim
    end

    # add flatten layer
    push!(encoder_chain, Flux.flatten)

    return Flux.Chain(encoder_chain...)
end


function small_decoder_backbone64x64(latent_dims::Int, out_channels::Int, rhidden_dims::Vector{Int}; activation = Flux.leakyrelu, in_dimensions = (16,16))

    decoder_chain = [Dense(latent_dims => rhidden_dims[1] * prod(in_dimensions)),
                     x -> reshape(x, in_dimensions[1], in_dimensions[2], rhidden_dims[1], :)]

    # add decoder layers
    for i in 1:length(rhidden_dims)-1
        push!(decoder_chain,
                ConvTranspose((4,4), rhidden_dims[i] => rhidden_dims[i + 1], activation, stride = 2, pad = SamePad())
        )
    end

    # add final layers
    push!(decoder_chain,
        ConvTranspose((4,4), rhidden_dims[end] => out_channels, activation; stride=2, pad = SamePad())
    )
    
    return Flux.Chain(decoder_chain...)
end