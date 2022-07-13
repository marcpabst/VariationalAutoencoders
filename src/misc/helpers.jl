function swapdims(A::AbstractArray, dimA::Int, dimB::Int)

    perm = ignore_derivatives() do
        perm = collect(1:length(size(A)))
        perm[dimA] = dimB
        perm[dimB] = dimA 
        return perm
    end

    return permutedims(A, perm)
end

function swapdims2(A::AbstractArray, dimA::Int, dimB::Int)
    perm = ntuple(ndims(A)) do d
       d==dimA ? dimB : d==dimB ? dimA : d
    end

    return permutedims(A, perm)
end


function flattenfirstdim(A::AbstractArray)
    new_size = Vector{Any}(collect(size(A)))
    A = reshape(A, 1, :, new_size[3:end]...)
    dropdims(A, dims = 1)
end

function appenddim(A::AbstractArray)
    Flux.unsqueeze(A; dims = length(size(A))+1)
end
