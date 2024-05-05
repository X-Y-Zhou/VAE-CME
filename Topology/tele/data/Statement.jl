# Topology/tele/data/matrix_tele.csv
# Topology/tele/data/matrix_telep0.csv
# Topology/tele/data/matrix_telep1.csv
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),30);rand(rng,Uniform(0.002,0.005),20)]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),30);rand(rng,Uniform(0.002,0.005),20)]
rho_on_list =    [rand(rng,Uniform(0.1,1),50);]
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topology/tele/data/params_matrix.csv",ps_matrix)

τ = 40
N = 80