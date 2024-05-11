# Topology/ps_bdv1.csv
# seed = 2 --> v2
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
ρ_list = [rand(rng,Uniform(1,5),50);rand(rng,Uniform(5,8),50);]
writedlm("Topologyv2/ps_bdv1.csv",ρ_list)

