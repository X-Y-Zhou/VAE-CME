# Topology/tele/data/matrix_telev1.csv
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),30);rand(rng,Uniform(0.002,0.005),20)]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),30);rand(rng,Uniform(0.002,0.005),20)]
rho_on_list =    [rand(rng,Uniform(0.1,1),50);]
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topology/tele/data/ps_telev1.csv",ps_matrix)

τ = 40
N = 80

# Topology/tele/data/matrix_telev2.csv
using Plots,Random,Distributions,DelimitedFiles
seed = 2
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),50);rand(rng,Uniform(0.002,0.005),50)]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),50);rand(rng,Uniform(0.002,0.005),50)]
rho_on_list =    [rand(rng,Uniform(0.1,1),100);]
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topology/tele/data/ps_telev2.txt",ps_matrix)

# Topology/tele/data/matrix_telev3.csv
using Plots,Random,Distributions,DelimitedFiles
seed = 2
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),30);rand(rng,Uniform(0.002,0.005),20)]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),30);rand(rng,Uniform(0.002,0.005),20)]

seed = 1
rng = Random.seed!(seed)
rho_on_list =    [rand(rng,Uniform(0.1,1),50);]
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topology/tele/data/ps_telev3.txt",ps_matrix)

# Topology/tele/data/matrix_telev4.csv
using Plots,Random,Distributions,DelimitedFiles
seed = 3
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),50);rand(rng,Uniform(0.002,0.005),50)]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),50);rand(rng,Uniform(0.002,0.005),50)]
rho_on_list =    [rand(rng,Uniform(0.1,1),100);]
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topology/tele/data/ps_telev4.txt",ps_matrix)

# Topology/tele/data/matrix_telev5.csv
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.1,0.5),50);rand(rng,Uniform(0.5,1),50)]
sigma_off_list = [rand(rng,Uniform(1,10),50);rand(rng,Uniform(5,15),50)]
rho_on_list =    [rand(rng,Uniform(2.5,6),100);]
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topology/tele/data/ps_telev5.txt",ps_matrix)


# Topology/tele/data/matrix_telev6.csv
using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.1,0.5),20);rand(rng,Uniform(0.5,1),20)]
sigma_off_list = [rand(rng,Uniform(1,10),20);rand(rng,Uniform(5,15),20)]
rho_on_list =    [rand(rng,Uniform(2.5,6),40);]
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]
ps_matrix = hcat([[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]...)
writedlm("Topology/tele/data/ps_telev6.txt",ps_matrix)