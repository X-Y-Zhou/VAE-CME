# Parallel
using Distributed,Pkg
addprocs(5)

# Import packages
@everywhere using Flux
@everywhere using Distributions, Distances,Random
@everywhere using DelimitedFiles, Plots
@everywhere include("../utils.jl")

# Define truncation
@everywhere N = 40

# Define kinetic parameters
@everywhere version = "train"
@everywhere ps_matrix = readdlm("Fig7/data/ps_$(version).txt")
@everywhere σu_list = ps_matrix[:,1]
@everywhere σb_list = ps_matrix[:,2]
@everywhere d_list = ps_matrix[:,3]
@everywhere batchsize = length(σu_list)

# Load training data 
@everywhere train_sol = readdlm("Fig7/data/SSA_prob_ps_$(version).txt")

# Model initialization
@everywhere latent_size = 2;
@everywhere encoder = Chain(Dense(2*N, 10,tanh),Dense(10, latent_size * 2));
@everywhere decoder = Chain(Dense(latent_size, 10),Dense(10 , N-1), x ->exp.(x));

@everywhere params1, re1 = Flux.destructure(encoder);
@everywhere params2, re2 = Flux.destructure(decoder);
@everywhere ps = Flux.params(params1,params2);

# Define the CME in the steady state
@everywhere function f_tele!(x,p1,p2,ϵ,σu,σb,d)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN = re2(p2)(z)
    return vcat(-σu*x[1] + d*x[2] + σb*x[N+1],
            [(-σu-(i-1)*d)*x[i] + i*d*x[i+1] + σb*x[i+N] for i in 2:N-1],
            (-σu-(N-1)*d)*x[N] + σb*x[2*N],

            σu*x[1] + (-σb-NN[1])*x[N+1] + d*x[N+2],
            [σu*x[i-N] + NN[i-N-1]*x[i-1] + (-σb-NN[i-N]-(i-N-1)*d)*x[i] + (i-N)*d*x[i+1] for i in (N+2):(2*N-1)],
            sum(x)-1)
end

# Define the CME solver
@everywhere sol_tele(p1,p2,ϵ,σu,σb,d,P_0) = nlsolve(x->f_tele!(x,p1,p2,ϵ,σu,σb,d),P_0).zero

@everywhere function solve_tele(σu,σb,d,p1,p2,ϵ)
    P_0_distribution = Poisson(σu/d)
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
    P_0_split = [P_0*σu/(σu+σb);P_0*σb/(σu+σb)]

    solution = sol_tele(p1,p2,ϵ,σu,σb,d,P_0_split)
    return solution
end

# Read trained VAE parameters
using CSV,DataFrames
df = CSV.read("Fig7/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

# Solve the CME
@everywhere ϵ = zeros(latent_size)
@time solution_tele = hcat(pmap(i->solve_tele(σu_list[i],σb_list[i],d_list[i],params1,params2,ϵ),1:batchsize)...)
solution_tele = solution_tele[1:N,:]+solution_tele[N+1:2*N,:]

# Plot probability distribution
function plot_distribution(set)
    plot(0:N-1,solution_tele[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix[set,:],digits=4)]),line=:dash)
end

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
plot_channel(1)

for i = 1:10
    p = plot_channel(i)
    savefig(p,"Fig7/results/fig_$i.svg")
end