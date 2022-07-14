module VariationalAutoencoders

export AbstractVariationalAutoencoder
export MSSIMVAE, BetaVEA 
export encode, decode, reconstruct, model_loss
export ImageNet64x64

include("models/abstract_variational_autoencoder.jl")

include("models/backbones.jl")
include("models/mssim_vae.jl")
include("models/beta_vae.jl")

include("datasets/imagenet64x64.jl")

end
