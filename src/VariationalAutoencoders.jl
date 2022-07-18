module VariationalAutoencoders



include("models/abstract_variational_autoencoder.jl")

#include("misc/distributions/vonmisesfischer.jl")

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
export ImageNet64x64, imagenet64x64_preprocess, imagenet64x64_reverse_preprocess, imagenet64x64_asimage
export Random64x64
export VonMisesFisher2


export default_encoder_backbone64x64, default_decoder_backbone64x64
export small_encoder_backbone64x64, small_decoder_backbone64x64


export ssim
export HyperSphericalUniform, KL, PowerSpherical, entropy


# make diff rule from ChainRules.jl known to ForwardDiff.jl
# NonconvexUtils.@ForwardDiff_frule SpecialFunctions.besselix(x1::ForwardDiff.Dual, x2::ForwardDiff.Dual)
# NonconvexUtils.@ForwardDiff_frule SpecialFunctions.besselyx(x1::ForwardDiff.Dual, x2::ForwardDiff.Dual)
# NonconvexUtils.@ForwardDiff_frule SpecialFunctions.bessely(x1::ForwardDiff.Dual, x2::ForwardDiff.Dual)

# NonconvexUtils.@ForwardDiff_frule SpecialFunctions.besselix(x1::Real, x2::ForwardDiff.Dual)
# NonconvexUtils.@ForwardDiff_frule SpecialFunctions.besselyx(x1::Real, x2::ForwardDiff.Dual)
# NonconvexUtils.@ForwardDiff_frule SpecialFunctions.bessely(x1::Real, x2::ForwardDiff.Dual)

end
