import DataLoaders.LearnBase: getobs, nobs
using FileIO, NPZ

"""
ImageNet dataset downsampled to to 64x64. 
Please note that images are loaded and kept (!) in memory.
"""
struct ImageNet64x64
    data::Array{UInt8, 4}
    labels::Vector{Int}
end

function ImageNet64x64(path::String) 
    # load file
    _f = FileIO.load(path)

    @assert size(_f["data"])[end] == size(_f["labels"],1) "Number of observations must match number of labels."

    return ImageNet64x64(_f["data"], _f["labels"])
end

nobs(data::ImageNet64x64) = size(data.data)[end]
getobs(data::ImageNet64x64, i) = @view data.data[:,:,:,i]

Base.getindex(data::ImageNet64x64, i::Int) = getobs(data, i) 
Base.length(data::ImageNet64x64) = nobs(data)