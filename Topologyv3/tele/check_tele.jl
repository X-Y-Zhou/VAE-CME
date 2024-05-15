# 仅用bursty的数据进行外拓
using Distributed,Pkg
addprocs(3)
# rmprocs(5)
nprocs()
workers()

@everywhere using Flux, DiffEqSensitivity, DifferentialEquations
@everywhere using Distributions, Distances,Random
@everywhere using DelimitedFiles, Plots

@everywhere include("../../utils.jl")

@everywhere τ = 100
@everywhere N = 120

# tele params and check_sol
@everywhere version = 1
@everywhere ps_matrix_tele = readdlm("Topologyv3/tele/data/ps_telev$version.txt")
@everywhere sigma_on_list = ps_matrix_tele[1,:]
@everywhere sigma_off_list = ps_matrix_tele[2,:]
@everywhere rho_on_list = ps_matrix_tele[3,:]
@everywhere rho_off = 0.0
@everywhere gamma= 0.0
@everywhere batchsize_tele = size(ps_matrix_tele,2)
check_sol1 = readdlm("Topologyv3/tele/data/matrix_tele_100-100.csv") # var = 0 Attribute = 0
check_sol2 = readdlm("Topologyv3/tele/data/matrix_tele_0-200.csv") # var = max Attribute = 1
check_sol3 = readdlm("Topologyv3/tele/data/matrix_tele_50-150.csv") # Attribute = 0.5

# model initialization
@everywhere latent_size = 2;
@everywhere encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
@everywhere decoder_1 = Chain(Dense(latent_size+1, 10),Dense(10 , N-1),x -> 0.03.* x.+[i/τ  for i in 1:N-1],x ->relu.(x));
@everywhere decoder_2 = Chain(decoder_1[1],decoder_1[2],decoder_1[3],decoder_1[4]);

@everywhere params1, re1 = Flux.destructure(encoder);
@everywhere params2, re2_1 = Flux.destructure(decoder_1);
@everywhere       _, re2_2 = Flux.destructure(decoder_2);
@everywhere ps = Flux.params(params1,params2);

# CME
@everywhere function f_tele!(x,p1,p2,ϵ,sigma_on,sigma_off,rho_on,Attribute)
    h = re1(p1)(x[1:N])
    # h = re1(p1)(x[1:N])*sigma_off/(sigma_on+sigma_off)
    # h = re1(p1)(x[1:N]*sigma_off/(sigma_on+sigma_off))

    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,Attribute)
    NN1 = re2_1(p2)(z)

    # NN1 = NN1*sigma_off/(sigma_on+sigma_off)
    # NN1 = NN1*(sigma_on+sigma_off)/sigma_off

    h = re1(p1)(x[N+1:2*N])
    # h = re1(p1)(x[N+1:2*N])*sigma_on/(sigma_on+sigma_off)
    # h = re1(p1)(x[N+1:2*N]*sigma_on/(sigma_on+sigma_off))
    
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,Attribute)
    NN2 = re2_1(p2)(z)
    
    # NN2 = NN2*sigma_on/(sigma_on+sigma_off)
    # NN2 = NN2*(sigma_on+sigma_off)/sigma_on

    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)
end

@everywhere sol_tele(p1,p2,ϵ,sigma_on,sigma_off,rho_on,P_0,Attrtibute) = nlsolve(x->f_tele!(x,p1,p2,ϵ,sigma_on,sigma_off,rho_on,Attrtibute),P_0).zero

# P0,P1
@everywhere function solve_tele(sigma_on,sigma_off,rho_on,p1,p2,ϵ,Attrtibute)
    P_0_distribution = Poisson(rho_on*τ*sigma_on)
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
    P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]

    solution = sol_tele(p1,p2,ϵ,sigma_on,sigma_off,rho_on,P_0_split,Attrtibute)
    solution = solution[1:N]+solution[N+1:2*N]
    return solution
end

using CSV,DataFrames
df = CSV.read("Topologyv3/vae/params_trained_vae_tele_bursty_better.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

Attrtibute = 0
@time solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
mse_tele = Flux.mse(solution_tele,check_sol1)

Attrtibute = 1
check_sol = check_sol2
@time solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
mse_tele = Flux.mse(solution_tele,check_sol)

function plot_distribution(set)
    plot(0:N-1,solution_tele[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,check_sol[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_tele[:,set],digits=4),Attrtibute]),line=:dash)
end
plot_distribution(30)

# function plot_distribution(set)
#     plot(0:N-1,solution_bursty1[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
#     plot!(0:N-1,train_sol1[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_bursty[:,set],digits=3)]),line=:dash)
# end

function plot_channel(i)
    p1 = plot_distribution(1+10*(i-1))
    p2 = plot_distribution(2+10*(i-1))
    p3 = plot_distribution(3+10*(i-1))
    p4 = plot_distribution(4+10*(i-1))
    p5 = plot_distribution(5+10*(i-1))
    p6 = plot_distribution(6+10*(i-1))
    p7 = plot_distribution(7+10*(i-1))
    p8 = plot_distribution(8+10*(i-1))
    p9 = plot_distribution(9+10*(i-1))
    p10 = plot_distribution(10+10*(i-1))
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,layouts=(2,5),size=(1500,600))
end
plot_channel(4)

for i = 1:5
    p = plot_channel(i)
    savefig(p,"Topologyv3/topo_results/fig_$i.svg")
end

set = 90
plot(0:N-1,solution_tele[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability",ylims=(-0.001,0.4))
plot!(0:N-1,check_sol[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_tele[:,set],digits=4)]),line=:dash)


Attrtibute = 0
check_sol = check_sol1
@time solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
mse_tele = Flux.mse(solution_tele,check_sol)
