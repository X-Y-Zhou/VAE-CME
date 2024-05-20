# 仅用bursty的数据进行外拓
using Distributed,Pkg
addprocs(3)
# rmprocs(5)
nprocs()
workers()

@everywhere using Flux, DiffEqSensitivity, DifferentialEquations
@everywhere using Distributions, Distances,Random
@everywhere using DelimitedFiles, Plots

@everywhere include("../utils.jl")

@everywhere τ = 10
@everywhere N = 120

# bursty params and train_sol
@everywhere ps_matrix_bursty = readdlm("Topologyv6/ps_burstyv1.csv")
@everywhere batchsize_bursty = size(ps_matrix_bursty,2)
@everywhere a_list = ps_matrix_bursty[1,:]
@everywhere b_list = ps_matrix_bursty[2,:]

train_sol1 = readdlm("Topologyv6/bursty_data/matrix_bursty_10-10.csv") # var = 0
train_sol2 = readdlm("Topologyv6/bursty_data/matrix_bursty_0-20.csv") # var = 1

train_sol3 = readdlm("Topologyv6/bursty_data/matrix_bursty_5-15.csv") # var = 0.5


# model initialization
@everywhere latent_size = 2;
@everywhere encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
@everywhere decoder_1 = Chain(Dense(latent_size+1, 10),Dense(10 , N-1),x -> x.+[i/τ  for i in 1:N-1],x ->relu.(x));
@everywhere decoder_2 = Chain(decoder_1[1],decoder_1[2],decoder_1[3],decoder_1[4]);

@everywhere params1, re1 = Flux.destructure(encoder);
@everywhere params2, re2_1 = Flux.destructure(decoder_1);
@everywhere       _, re2_2 = Flux.destructure(decoder_2);
@everywhere ps = Flux.params(params1,params2);

# CME
@everywhere function f_extend!(x,p1,p2,ϵ,a,b,Attrtibute)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,Attrtibute)
    NN = re2_1(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

@everywhere sol_bursty_extend(p1,p2,ϵ,a,b,P0,Attrtibute) = nlsolve(x->f_extend!(x,p1,p2,ϵ,a,b,Attrtibute),P0).zero

@everywhere function solve_bursty_extend(a,b,p1,p2,ϵ,Attrtibute)
    P_0_distribution = NegativeBinomial(a*τ, 1/(1+b))
    P_0 = [pdf(P_0_distribution,i) for i=0:N-1]
    solution = sol_bursty_extend(p1,p2,ϵ,a,b,P_0,Attrtibute)
    return solution
end

using CSV,DataFrames
df = CSV.read("Topologyv6/vae/params_trained_vae_tele_bursty.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

ϵ = zeros(latent_size)
Attrtibute = 0.5
check_sol = train_sol3
@time solution_bursty_extend = hcat(pmap(i->solve_bursty_extend(a_list[i],b_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_bursty)...);
mse_bursty2 = Flux.mse(solution_bursty_extend,check_sol)

function plot_distribution(set)
    plot(0:N-1,solution_bursty_extend[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,check_sol[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_bursty[:,set],digits=4)]),line=:dash)
end
plot_distribution(30)

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
plot_channel(3)

for i = 1:5
    p = plot_channel(i)
    savefig(p,"Topologyv6/topo_results/fig_$i.svg")
end


