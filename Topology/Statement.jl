# Topology/ps_bdv1.csv
# seed = 2 --> v2
using Plots,Random,Distributions,DelimitedFiles
seed = 2
rng = Random.seed!(seed)
ρ_list = rand(rng,Uniform(0.1,1),100)
τ = 40
N = 80
matrix_bd = hcat([birth_death(N, ρ_list[i], τ) for i = 1:length(ρ_list)]...)
writedlm("Topology/ps_bdv2.csv",ρ_list)

