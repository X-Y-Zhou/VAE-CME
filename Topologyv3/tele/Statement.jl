# Topology/tele/data/matrix_telev1.csv
# seed = 2 --> v2
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),10);rand(rng,Uniform(0.002,0.005),10);
                  rand(rng,Uniform(0.01,0.05),10);
                  rand(rng,Uniform(0.1,0.5),10);rand(rng,Uniform(0.5,1),10);]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),10);rand(rng,Uniform(0.002,0.005),10);
                  rand(rng,Uniform(0.1,0.5),10);
                  rand(rng,Uniform(1,5),10);rand(rng,Uniform(4,8),10);]
rho_on_list =    [rand(rng,Uniform(0.1,0.5),20);
                  rand(rng,Uniform(0.8,1.5),10);
                  rand(rng,Uniform(1,5),20);]
τ = 100
N = 120
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topologyv3/tele/data/ps_telev1.txt",ps_matrix)