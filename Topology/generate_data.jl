using Plots,Random,Distributions
using DelimitedFiles

include("../utils.jl")

seed = 1
rng = Random.seed!(seed)
ρ_list = rand(rng,Uniform(0.1,1),60)
τ = 40
N = 80
matrix_bd = hcat([birth_death(N, ρ_list[i], τ) for i = 1:length(ρ_list)]...)
writedlm("Topology/matrix_bd.csv",matrix_bd)

train_sol = readdlm("Topology/matrix_bd.csv")



