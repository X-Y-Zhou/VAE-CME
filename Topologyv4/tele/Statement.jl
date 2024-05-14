# Topology/tele/data/matrix_telev1.csv
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),10);rand(rng,Uniform(0.002,0.005),10);
                  rand(rng,Uniform(0.01,0.05),10);
                  rand(rng,Uniform(0.1,0.5),10);rand(rng,Uniform(0.5,1),10);]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),10);rand(rng,Uniform(0.002,0.005),10);
                  rand(rng,Uniform(0.1,0.5),10);
                  rand(rng,Uniform(1,5),10);rand(rng,Uniform(4,8),10);]
rho_on_list =    [rand(rng,Uniform(0.5,1),20);
                  rand(rng,Uniform(0.5,2),10);
                  rand(rng,Uniform(3,8),20);]
τ = 50
N = 120
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topologyv4/tele/data/ps_telev1.txt",ps_matrix)

# Topology/tele/data/matrix_telev2.csv
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),10);rand(rng,Uniform(0.002,0.005),10);
                  rand(rng,Uniform(0.01,0.05),10);
                  rand(rng,Uniform(0.1,0.5),10);rand(rng,Uniform(0.5,1),10);]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),10);rand(rng,Uniform(0.002,0.005),10);
                  rand(rng,Uniform(0.1,0.5),10);
                  rand(rng,Uniform(1,5),10);rand(rng,Uniform(4,8),10);]
rho_on_list =    [rand(rng,Uniform(1,5),20);
                  rand(rng,Uniform(1,5),10);
                  rand(rng,Uniform(10,20),20);]
τ = 10
N = 120
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topologyv4/tele/data/ps_telev2.txt",ps_matrix)





# Topology/tele/data/matrix_telev1.csv
# seed = 2 --> v2
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),20);rand(rng,Uniform(0.002,0.005),20);
                  rand(rng,Uniform(0.1,0.5),20);
                  rand(rng,Uniform(0.5,1),20);rand(rng,Uniform(0.5,1),20)]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),20);rand(rng,Uniform(0.002,0.005),20);
                  rand(rng,Uniform(0.1,0.5),20);
                  rand(rng,Uniform(0.5,1),20);rand(rng,Uniform(0.5,1),20)]
# rho_on_list =    [rand(rng,Uniform(0.5,1),50);rand(rng,Uniform(1,5),50);]
rho_on_list =    [rand(rng,Uniform(1,5),50);rand(rng,Uniform(5,8),50);]
τ = 10
N = 120
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topologyv2/tele/data/ps_telev2.txt",ps_matrix)

# Topology/tele/data/matrix_telev3.csv
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.1,0.5),25);rand(rng,Uniform(0.5,1),25);]
# sigma_off_list = [rand(rng,Uniform(1,5),50);]
sigma_off_list = [rand(rng,Uniform(1,7),25);rand(rng,Uniform(4,10),25);]
rho_on_list =    [rand(rng,Uniform(10,15),25);rand(rng,Uniform(15,25),25);]
τ = 10
N = 120
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topologyv2/tele/data/ps_telev3.txt",ps_matrix)