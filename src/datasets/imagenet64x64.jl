import DataLoaders.LearnBase: getobs, nobs
using FileIO
using Statistics
using ImageCore

const imagenet_means = [0.485, 0.456, 0.406] 
const imagenet_stds  = [0.229, 0.224, 0.225]

"""
ImageNet dataset downsampled to to 64x64. 
Please note that images are loaded and kept (!) in memory.
"""
struct ImageNet64x64{T <: Integer}
    data::Array{T, 4}
    labels::Vector{Int}
    datatype::Type{U}
    normalize::Bool
end

function ImageNet64x64(path::String; datatype::Type = Float32, demean::Bool = true, normalize::Bool = true) 
    # load file
    _f = FileIO.load(path)

    return ImageNet64x64(_f["data"], _f["labels"], datatype, normalize)
end

# for DataLoaders.DataLoader
nobs(data::ImageNet64x64) = size(data.data)[end]

getobs(data::ImageNet64x64, i) = @view data.data[:,:,:,i]

# for MLUtils.DataLoader
Base.getindex(data::ImageNet64x64, i) = getobs(data, i) 
Base.length(data::ImageNet64x64) = nobs(data)

function imagenet64x64_preprocess(data::Array{T, 4}; demean = true, normalize = true) where T <: Integer
    out = convert(Float32, data) ./ 255 
    out .= demean ? out .- imagenet_means : out
    out .= normalize ? out ./ imagenet_stds : out

    return out
end

function imagenet64x64_reverse_preprocess(data::Array{Float32, 4}; demean = true, normalize = true)
    out .= normalize ? out .* imagenet_stds : out
    out .= demean ? out .+ imagenet_means : out
    return out
end

function imagenet64x64_asimage(data::Array{Real, 3})
    out = convert(Float32, data) ./ 255 
    return colorview(RGB, permutedims(out, (3,2,1)))
end
