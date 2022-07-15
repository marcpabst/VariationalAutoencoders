import DataLoaders.LearnBase: getobs, nobs
using FileIO
using Statistics

"""
Random data.
"""
struct Random64x64{T}
    data::Array{T, 4}
end

function Random64x64(nobs::Int; n_channels::Int = 1, datatype = Float32) 
    return Random64x64( rand(datatype, 64, 64, n_channels, nobs) )
end

# for DataLoaders.DataLoader
nobs(data::Random64x64) = size(data.data)[end]
getobs(data::Random64x64, i) = @view data.data[:,:,:,i]

# for MLUtils.DataLoader
Base.getindex(data::Random64x64, i) = getobs(data, i) 
Base.length(data::Random64x64) = nobs(data)