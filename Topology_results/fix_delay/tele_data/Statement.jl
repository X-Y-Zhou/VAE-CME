# Topology_results/fix_delay/tele_data/matrix_telev1.txt
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),20);rand(rng,Uniform(0.002,0.005),20);
                  rand(rng,Uniform(0.1,0.5),20);
                  rand(rng,Uniform(0.5,1),20);rand(rng,Uniform(0.5,1),20)]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),20);rand(rng,Uniform(0.002,0.005),20);
                  rand(rng,Uniform(0.1,0.5),20);
                  rand(rng,Uniform(0.5,1),20);rand(rng,Uniform(0.5,1),20)]
rho_on_list =    [rand(rng,Uniform(1,5),50);rand(rng,Uniform(5,8),50);]
τ = 10
N = 120
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
# writedlm("Topology_results/fix_delay/tele_data/matrix_telev1.txt",ps_matrix)

# Topology_results/fix_delay/tele_data/matrix_telev3.txt
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.1,0.5),25);rand(rng,Uniform(0.5,1),25);]
sigma_off_list = [rand(rng,Uniform(1,7),25);rand(rng,Uniform(4,10),25);]
rho_on_list =    [rand(rng,Uniform(10,15),25);rand(rng,Uniform(15,25),25);]
τ = 10
N = 120
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
# writedlm("Topology_results/fix_delay/tele_data/matrix_telev3.txt",ps_matrix)

# Topology_results/fix_delay/tele_data/ps_televfinal.txt
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),20);rand(rng,Uniform(0.002,0.005),20);
                  rand(rng,Uniform(0.1,0.5),20);
                  rand(rng,Uniform(0.5,1),20);rand(rng,Uniform(0.5,1),20);
                  rand(rng,Uniform(0.1,0.5),25);rand(rng,Uniform(0.5,1),25);]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),20);rand(rng,Uniform(0.002,0.005),20);
                  rand(rng,Uniform(0.1,0.5),20);
                  rand(rng,Uniform(0.5,1),20);rand(rng,Uniform(0.5,1),20);
                  rand(rng,Uniform(1,7),25);rand(rng,Uniform(4,10),25);]
rho_on_list =    [rand(rng,Uniform(1,5),50);rand(rng,Uniform(5,8),50);
                  rand(rng,Uniform(10,15),25);rand(rng,Uniform(15,25),25);]
τ = 10
N = 120
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topology_results/fix_delay/tele_data/ps_televfinal.txt",ps_matrix)

