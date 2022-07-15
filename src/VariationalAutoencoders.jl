module VariationalAutoencoders

export AbstractVariationalAutoencoder
export MSSIMVAE, BetaVEA, HypersphericalVAE, VanillaVAE
export encode, decode, reconstruct, model_loss
export train!
export ImageNet64x64


export HyperSphericalUniform, KL

include("models/abstract_variational_autoencoder.jl")

include("models/backbones.jl")
include("models/mssim_vae.jl")
include("models/vanilla_vae.jl")
include("models/hyperspherical_vae.jl")

include("datasets/imagenet64x64.jl")

include("training/train.jl")

end
