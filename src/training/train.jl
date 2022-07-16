using Flux
using ProgressMeter
using VariationalAutoencoders
using MLUtils
using DataFrames

"""
Default training loop for VAEs.
"""
function train!(model::AbstractVariationalAutoencoder, training_data, args; 
                start_epoch = 1, 
                logdf = DataFrame(epoch = Int[], loss = Float64[], KL_loss = Float64[], recon_loss = Float64[]),
                cb::Union{Function, Nothing} = nothing,
                pp::Union{Function, Nothing} = nothing)

    @assert args[:epochs] >= start_epoch "Model fully trained."

    @info "Starting Training..." typeof(model) start_epoch args[:epochs] args[:η]

    opt = ADAM(args[:η])
    ps = Flux.params(model)
    
    for epoch in start_epoch:args[:epochs]

        _loss, _loss_KL, _loss_recon = 0., 0., 0.

        data_loader = MLUtils.DataLoader(training_data; batchsize=args[:batchsize], shuffle=true)

        p = Progress(length(training_data), desc = "Epoch $epoch")

        for x in data_loader
            x = pp === nothing ? x : pp(x)
            x = x |> gpu
            
            gs = gradient(ps) do
                loss = model_loss(model, x)
                _loss = loss[1]
                _loss_KL = loss[:loss_KL]
                _loss_recon = loss[:loss_recon]
                return _loss
            end
    
            Flux.update!(opt, ps, gs)

            next!(p; step=size(x)[end])

        end

        # push loss to logdf
        push!(logdf, [epoch, 
            first(_loss), 
            first(_loss_KL), 
            first(_loss_recon)
            ]);

        # call callback
        if cb !== nothing
            cb(model, logdf, epoch);
        end
    end

    return logdf
end
