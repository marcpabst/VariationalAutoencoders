import DataLoaders.LearnBase: getobs, nobs
using FileIO

struct ImageFilesDataset
    files::Vector{String}
end

function ImageFilesDataset(folder::String) 
    files = String[]

    for e in readdir(folder, join=true)
        if isfile(e)
            push!(files, e)
        end
    end

    return ImageFilesDataset(files)
end

nobs(data::ImageFilesDataset) = length(data.files)
getobs(data::ImageFilesDataset, i::Int) = FileIO.load(data.files[i])


Base.getindex(data::ImageNet, i::Int) = getobs(data, i) 
Base.length(data::ImageNet) = nobs(data)