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
end

function ImageNet64x64(path::String; demean::Bool = true) 
    # load file
    _f = FileIO.load(path)

    return ImageNet64x64(_f["data"], _f["labels"])
end

# for DataLoaders.DataLoader
nobs(data::ImageNet64x64) = size(data.data)[end]

getobs(data::ImageNet64x64, i) = @view data.data[:,:,:,i]

# for MLUtils.DataLoader
Base.getindex(data::ImageNet64x64, i) = getobs(data, i) 
Base.length(data::ImageNet64x64) = nobs(data)

function imagenet64x64_preprocess(data; demean = true, normalize = true)
    out = Base.convert.(Float32, data) ./ 255 
    
    if demean
        out[:,:,1,:] .= out[:,:,1,:] .- imagenet_means[1]
        out[:,:,2,:] .= out[:,:,2,:] .- imagenet_means[2]
        out[:,:,3,:] .= out[:,:,3,:] .- imagenet_means[3]
    end

    if normalize 
        out[:,:,1,:] .= out[:,:,1,:] ./ imagenet_stds[1]
        out[:,:,2,:] .= out[:,:,2,:] ./ imagenet_stds[2]
        out[:,:,3,:] .= out[:,:,3,:] ./ imagenet_stds[3]
    end
    
    return out
end

function imagenet64x64_reverse_preprocess(data; demean = true, normalize = true)

    if normalize 
        out[:,:,1,:] .= out[:,:,1,:] .* imagenet_stds[1] # R
        out[:,:,2,:] .= out[:,:,2,:] .* imagenet_stds[2] # G
        out[:,:,3,:] .= out[:,:,3,:] .* imagenet_stds[3] # B
    end
    
    if demean
        out[:,:,1,:] .= out[:,:,1,:] .+ imagenet_means[1] # R
        out[:,:,2,:] .= out[:,:,2,:] .+ imagenet_means[2] # G
        out[:,:,3,:] .= out[:,:,3,:] .+ imagenet_means[3] # B
    end

    return out
end

function imagenet64x64_asimage(data)
    out = Base.convert.(Float32, data) ./ 255 
    return colorview(RGB, permutedims(out, (3,2,1)))
end
