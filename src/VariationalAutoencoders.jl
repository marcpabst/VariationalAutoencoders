module VariationalAutoencoders

export AbstractVariationalAutoencoder
export MSSIMVAE, BetaVEA 
export encode, decode, reconstruct, model_loss

include("models/backbones.jl")
include("models/mssim_vae.jl")
include("models/beta_vae.jl")

end
