using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
a_list = [rand(rng,Uniform(0.1,0.5),25);rand(rng,Uniform(0.5,1),25);]
b_list = [rand(rng,Uniform(1,3),25);rand(rng,Uniform(1.5,5),25)]
ps_matrix = hcat([[a_list[i],b_list[i]] for i=1:length(a_list)]...)
τ = 10
N = 120
writedlm("Topology_results/var_delay/bursty_data/ps_burstyv1.txt",ps_matrix)

