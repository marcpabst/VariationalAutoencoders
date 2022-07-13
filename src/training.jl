using Flux
using ProgressMeter
using VariationalAutoencoders

function train!(model::AbstractVariationalAutoencoder, training_data, args; start_epoch = 1)
    @info "Start Training, total $(args[:epochs]) epochs"

    @assert args[:epochs] > start_epoch "Model fully trained."

    logdf = DataFrame(epoch = Int[], loss = Float64[], KL_loss = Float64[], recon_loss = Float64[])

    opt = ADAM(args[:η])
    ps = Flux.params(model)
    
    @showprogress for epoch in start_epoch:args[:epochs]

        _loss, _loss_KL, _loss_recon = 0., 0., 0.

        for (x, _) in training_data

            x = reshape(x, 64,64,1,:) |> gpu
            
            gs = gradient(ps) do
                loss = model_loss(model, x)
                _loss = loss[1]
                _loss_KL = loss[:loss_KL]
                _loss_recon = loss[:loss_recon]
                return _loss
            end

            Flux.update!(opt, ps, gs)
        end

     # push loss to logdf
     push!(logdf, [epoch, 
        first(_loss), 
        first(_loss_KL), 
        first(_loss_recon)
        ]);

    end

    return logdf
end
