module VariationalAutoencoders


include("models/abstract_variational_autoencoder.jl")

# include("misc/distributions/hyperspherical_uniform.jl")
include("misc/distributions/nonmutating.jl")

include("models/backbones.jl")
include("models/mssim_vae.jl")
include("models/vanilla_vae.jl")
include("models/beta_vae.jl")
include("models/hyperspherical_vae.jl")
include("models/producthyperspherical_vae.jl")

include("datasets/imagenet64x64.jl")
include("datasets/random64x64.jl")

include("training/train.jl")

export AbstractVariationalAutoencoder
export MSSIMVAE, BetaVEA, HypersphericalVAE, VanillaVAE, ProductHypersphericalVAE 
export encode, decode, reconstruct, model_loss
export train!
export ImageNet64x64, imagenet64x64_preprocess, imagenet64x64_reverse_preprocess, imagenet64x64_asimage
export Random64x64
export VonMisesFisher2


export default_encoder_backbone64x64, default_decoder_backbone64x64
export small_encoder_backbone64x64, small_decoder_backbone64x64

export rrand

export ssim

# function trand(rng::AbstractRNG, spl::Distributions.PowerSphericalSampler)
#     z = rand(rng, spl.dist_b)
#     v = rand(rng, spl.dist_u)

#     t = 2 * z - 1
#     m = sqrt(1 - t ^ 2) * v'

#     y = [t; m]
#     e_1 = [1.; zeros(eltype(spl.μ), length(spl) -1)]

#     û = e_1 - spl.μ
#     u = normalize(û)

#     return (-1) * (I(length(spl)) .- 2*u*u') * y
# end

# trand(s::PowerSpherical) = trand(Random.GLOBAL_RNG, Distributions.sampler(s))

# export trand
end
