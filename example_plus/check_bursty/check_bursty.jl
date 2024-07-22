# Parallel
using Distributed,Pkg
addprocs(3)

# Import packages
@everywhere using Flux, DiffEqSensitivity, DifferentialEquations
@everywhere using Distributions, Distances,Random
@everywhere using DelimitedFiles, Plots
@everywhere include("../../utils.jl")

# Define time delay and truncation
@everywhere τ = 30
@everywhere N = 120

# Define kinetic parameters
@everywhere ps_matrix_bursty = readdlm("example_plus/check_bursty/ps_bursty_check.txt")
@everywhere batchsize_bursty = size(ps_matrix_bursty,2)
@everywhere a_list = ps_matrix_bursty[1,:]
@everywhere b_list = ps_matrix_bursty[2,:]

# Load check data 
check_sol1 = readdlm("example_plus/check_bursty/Exact_proba/bursty_exact_Attr=0.0.txt") # Attrtibute = 0   τ ~ Normal(30,0)
check_sol2 = readdlm("example_plus/check_bursty/Exact_proba/bursty_exact_Attr=0.5.txt")  # Attrtibute = 0.5    τ ~ Normal(30,5)
check_sol3 = readdlm("example_plus/check_bursty/Exact_proba/bursty_exact_Attr=1.0.txt")  # Attrtibute = 1.0  τ ~ Normal(30,10)

# Model initialization
@everywhere latent_size = 2;
@everywhere encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
@everywhere decoder_1 = Chain(Dense(latent_size+1, 10),Dense(10 , N-1),x -> x.+[i/τ  for i in 1:N-1],x ->relu.(x));
@everywhere decoder_2 = Chain(decoder_1[1],decoder_1[2],decoder_1[3],decoder_1[4]);

@everywhere params1, re1 = Flux.destructure(encoder);
@everywhere params2, re2_1 = Flux.destructure(decoder_1);
@everywhere       _, re2_2 = Flux.destructure(decoder_2);
@everywhere ps = Flux.params(params1,params2);

# Define the CME in the steady state
@everywhere function f_bursty!(x,p1,p2,ϵ,a,b,Attribute)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,Attribute)
    NN = re2_1(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

# Define the CME solver
@everywhere sol_bursty(p1,p2,ϵ,a,b,P_0,Attrtibute) = nlsolve(x->f_bursty!(x,p1,p2,ϵ,a,b,Attrtibute),P_0).zero
@everywhere function solve_bursty(a,b,p1,p2,ϵ,Attrtibute)
    P_0_distribution = NegativeBinomial(a*τ, 1/(1+b))
    P_0 = [pdf(P_0_distribution,i) for i=0:N-1]

    solution = sol_bursty(p1,p2,ϵ,a,b,P_0,Attrtibute)
    return solution
end

# Read trained VAE parameters
using CSV,DataFrames
df = CSV.read("example_plus/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]

# Solve the CME
ϵ = zeros(latent_size)
Attrtibute = 0
@time solution_bursty1 = hcat(pmap(i->solve_bursty(a_list[i],b_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_bursty)...);
mse_bursty1 = Flux.mse(solution_bursty1,check_sol1)

Attrtibute = 0.5
@time solution_bursty2 = hcat(pmap(i->solve_bursty(a_list[i],b_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_bursty)...);
mse_bursty2 = Flux.mse(solution_bursty2,check_sol2)

Attrtibute = 1.0
@time solution_bursty3 = hcat(pmap(i->solve_bursty(a_list[i],b_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_bursty)...);
mse_bursty3 = Flux.mse(solution_bursty3,check_sol3)

# Plot probability distribution
function plot_distribution(set)
    plot(0:N-1,solution_bursty1[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,check_sol1[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_bursty[:,set],digits=4)]),line=:dash)
end

function plot_channel()
    p1 = plot_distribution(1)
    p2 = plot_distribution(2)
    p3 = plot_distribution(3)
    p4 = plot_distribution(4)
    p5 = plot_distribution(5)
    p6 = plot_distribution(6)
    plot(p1,p2,p3,p4,p5,p6,layouts=(3,2),size=(600,900))
end
plot_channel()
# savefig("example_plus/check_bursty/results/Attri=0.0.svg")

# writedlm("example_plus/check_bursty/pre_proba/bursty_pre_Attr=0.0.txt",solution_bursty1)
# writedlm("example_plus/check_bursty/pre_proba/bursty_pre_Attr=0.5.txt",solution_bursty2)
# writedlm("example_plus/check_bursty/pre_proba/bursty_pre_Attr=1.0.txt",solution_bursty3)




