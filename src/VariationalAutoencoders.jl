module VariationalAutoencoders

include("models/abstract_variational_autoencoder.jl")

include("misc/distributions/vonmisesfischer.jl")

include("models/backbones.jl")
include("models/mssim_vae.jl")
include("models/vanilla_vae.jl")
include("models/beta_vae.jl")
include("models/hyperspherical_vae.jl")

include("datasets/imagenet64x64.jl")
include("datasets/random64x64.jl")

include("training/train.jl")

export AbstractVariationalAutoencoder
export MSSIMVAE, BetaVEA, HypersphericalVAE, VanillaVAE
export encode, decode, reconstruct, model_loss
export train!
export ImageNet64x64, Random64x64
export VonMisesFisher2


export default_encoder_backbone64x64, default_decoder_backbone64x64
export small_encoder_backbone64x64, small_decoder_backbone64x64


export ssim
export HyperSphericalUniform, KL

end
