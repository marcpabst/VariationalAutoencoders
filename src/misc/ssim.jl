using Flux
using Statistics
using MLUtils
using ChainRulesCore

include("helpers.jl")

"""Create 1-D gauss kernel"""
function _fspecial_gauss_1d(siz, sigma)
    coords = range(0., siz-1; step=1)
    coords = coords .- siz ÷ 2

    g = exp.(-(coords .^ 2) ./ (2 .* sigma .^ 2))
    g = g / sum(g)

    return appenddim(appenddim(g))
end

"""Blur input with 1-D kernel"""
function gaussian_filter(input, win)
    out = input
    C = size(out, 3)

    for (i, s) in enumerate(size(out)[1:2])
        if s >= size(win)[1]
            out = NNlib.conv(out, swapdims2(win, length(size(win)) - (1+i), 1); stride=1, pad=0, groups=C)
        else
            println("WARNING: Skipping Gaussian Smoothing at dimension 2+$(i) for input: $(size(input)) and win size:  $(size(win)[1])")
        end
    end

    return out
end

function _ssim(X, Y, data_range, win; size_average=true, K=(0.01, 0.03))
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 .* data_range) ^ 2
    C2 = (K2 .* data_range) ^ 2

    win = win

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1 .^ 2
    mu2_sq = mu2 .^ 2
    mu1_mu2 = mu1 .* mu2

    sigma1_sq = compensation .* (gaussian_filter(X .* X, win) .- mu1_sq)
    sigma2_sq = compensation .* (gaussian_filter(Y .* Y, win) .- mu2_sq)
    sigma12 = compensation .* (gaussian_filter(X .* Y, win) .- mu1_mu2)

    cs_map = (2 .* sigma12 .+ C2) ./ (sigma1_sq .+ sigma2_sq .+ C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 .* mu1_mu2 .+ C1) ./ (mu1_sq .+ mu2_sq .+ C1)) .* cs_map

    ssim_per_channel = flattenfirstdim(ssim_map)

    ssim_per_channel = mean(ssim_per_channel, dims = 1)
    
    cs = flattenfirstdim(cs_map)
    cs = mean(cs, dims = 1)

    return ssim_per_channel, cs
end

function ssim(X, Y; data_range=255, size_average=true, win_size=11, win_sigma=1.5, win=nothing, K=(0.01, 0.03), nonnegative_ssim=false)
        
    if win !== nothing  # set win_size
        win_size = size(win, 1)
    end

    #assert win_size % 2 == 1 "Window size should be odd."

    if win === nothing
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = repeat(win, [size(X,3); repeat([1], length(size(X)) - 1 )]...)
    end

    ssim_per_channel, _ = _ssim(X, Y, data_range, win; size_average=false, K=K)

    if nonnegative_ssim
        ssim_per_channel = Flux.relu.(ssim_per_channel)
    end

    if size_average
        return mean(ssim_per_channel)
    else
        return mean(ssim_per_channel, dims=2)
    end
end


function msssim(X, Y; data_range=255, size_average=true, win_size=11, win_sigma=1.5, win=nothing, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], K=(0.01, 0.03))

    if win !== nothing  # set win_size
        win_size = size(win, 1)
    end

    #assert win_size % 2 == 1 "Window size should be odd."

    if win === nothing
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = repeat(win, [size(X,3); repeat([1], length(size(X)) - 1 )]...)
    end

    levels = size(weights,1)
    ssim_per_channel, cs = _ssim(X, Y, data_range, win; size_average=false, K=K)
    mcs = []

    for i in 1:levels
        if i < levels
            push!(mcs, Flux.relu.(cs))
            padding = tuple([s % 2 for s in size(X)[1:end-2]]...)
            X = Flux.meanpool(X, (2,2), pad=padding)
            Y = Flux.meanpool(Y, (2,2), pad=padding)
        end
    end

    
    ssim_per_channel = Flux.relu.(ssim_per_channel)

    #return mcs, ssim_per_channel

    #mcs_and_ssim = permutedims([mcs; ssim_per_channel],(3,2,1))  # (level, batch, channel)
    mcs_and_ssim = cat([mcs; [ssim_per_channel]]...; dims = 4)[1,:,:,:]
    
    ms_ssim_val =  prod(mcs_and_ssim .^ reshape(weights, 1, 1, :), dims=3)

    if size_average
        return mean(ms_ssim_val)
    else
        return mean(ms_ssim_val; dims = 2)
    end
end
