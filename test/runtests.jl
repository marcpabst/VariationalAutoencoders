using VariationalAutoencoders
using Test

@testset "VariationalAutoencoders.jl" begin
    # Write your tests here.
end


# KullbackLeibler(VonMises{Float64}(0.0, 1.0), Uniform(-pi, pi), HyperCube([-pi,pi])) == KL_div_stable(2,1) 