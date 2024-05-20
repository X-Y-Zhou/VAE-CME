# Topology/tele/data/matrix_telev1.csv
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),10);rand(rng,Uniform(0.002,0.005),10);
                    rand(rng,Uniform(0.1,0.5),10);rand(rng,Uniform(0.5,1),20);]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),10);rand(rng,Uniform(0.002,0.005),10);
                    rand(rng,Uniform(1,5),10);rand(rng,Uniform(4,10),20);]
rho_on_list =    [rand(rng,Uniform(1,5),10);rand(rng,Uniform(5,8),10);
                    rand(rng,Uniform(10,15),10);rand(rng,Uniform(15,30),20);]
τ = 10
N = 120
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topologyv6/tele/data/ps_telev1.txt",ps_matrix)


# Topology/tele/data/matrix_telev2.csv
using Plots,Random,Distributions,DelimitedFiles
seed = 2
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),10);rand(rng,Uniform(0.003,0.007),10);]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),10);rand(rng,Uniform(0.003,0.007),10);]
rho_on_list =    [rand(rng,Uniform(1,5),10);rand(rng,Uniform(1,5),10);]
τ = 10
N = 120
batchsize = length(rho_on_list)
# index_list = 1:20
index_list = [1,2,7,8,9,11,12,13,17,19]
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i in index_list]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i in index_list]...)
writedlm("Topologyv6/tele/data/datav2/ps_telev2.txt",ps_matrix)






