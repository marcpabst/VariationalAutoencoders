using Flux
using ProgressMeter
using VariationalAutoencoders
using MLUtils
using Zygote
using CUDA
using Random
using DataFrames
using Statistics

"""
Default training loop for VAEs.
"""
function train!(model::AbstractVariationalAutoencoder, training_data, testing_data, args; 
                start_epoch = 1, 
                logdf = DataFrame(
                            epoch = Int[], 
                            loss = Float64[], 
                            KL_loss = Float64[], 
                            recon_loss = Float64[], 
                            kappa_mean = Float64[], 
                            kappa_var = Float64[]
                            ),
                cb::Union{Function, Nothing} = nothing,
                pp::Union{Function, Nothing} = nothing,
                seed = 42)

    Random.seed!(seed); CUDA.seed!(seed)

    @assert args[:epochs] >= start_epoch "Model fully trained."

    @info "Starting Training..." typeof(model) start_epoch args[:epochs] args[:η]

    opt = ADAM(args[:η])
    ps = Flux.params(model)
    
    @showprogress for epoch in start_epoch:args[:epochs]

        # keep track of loss
        _loss, _loss_KL, _loss_recon = 0., 0., 0.

        data_loader = MLUtils.DataLoader(training_data; batchsize=args[:batchsize], shuffle=true)

        #p = Progress(length(training_data), desc = "Epoch $epoch")

        for x in data_loader
            x = pp === nothing ? x : pp(x)
            x = x |> gpu

            loss = nothing

            gs = Zygote.gradient(ps) do
                loss = model_loss(model, x)
                return loss[:loss]
            end

            _loss = _loss + loss[:loss]
            _loss_KL = _loss_KL + loss[:loss_KL]
            _loss_recon = _loss_recon + loss[:loss_recon]
        
            Flux.update!(opt, ps, gs)

            #next!(p; step=size(x)[end])

        end

        # keep track of kappa from test_data after every epoch
        predictions = [reconstruct(model, testing_data[i:i]) for i in 1:length(testing_data)];
        kappas = cpu(first.(getindex.(predictions, 2)));

        # push loss to logdf
        push!(logdf, [epoch, 
            first(_loss), 
            first(_loss_KL), 
            first(_loss_recon),
            mean(kappas),
            var(kappas),
            ]);

        # call callback
        if cb !== nothing
            cb(model, logdf, epoch);
        end
    end

    return logdf
end
