import DataLoaders.LearnBase: getobs, nobs
using FileIO
using Statistics

"""
ImageNet dataset downsampled to to 64x64. 
Please note that images are loaded and kept (!) in memory.
"""
struct ImageNet64x64{T <: Integer, U}
    data::Array{T, 4}
    meanimage::Union{Array{U, 4}, Nothing}
    labels::Vector{Int}
    datatype::Type{U}
    demean::Bool
    normalize::Bool
end

function ImageNet64x64(path::String; datatype::Type = Float32, demean::Bool = true, normalize::Bool = true) 
    # load file
    _f = FileIO.load(path)

    # compute mean image if demean = true
    meanimage = demean ? convert.(datatype, mean(_f["data"], dims=4)  ./ 255 ) : nothing

    return ImageNet64x64(_f["data"], meanimage, _f["labels"], datatype, demean, normalize)
end

# for DataLoaders.DataLoader
nobs(data::ImageNet64x64) = size(data.data)[end]

function getobs(data::ImageNet64x64, i) 
    out = convert.(data.datatype, data.data[:,:,:,i])
    out .= data.normalize ? out ./ 255 : out
    out .= data.demean ? out .- data.meanimage : out
    return out
end

# for MLUtils.DataLoader
Base.getindex(data::ImageNet64x64, i) = getobs(data, i) 
Base.length(data::ImageNet64x64) = nobs(data)