# Parallel
using Distributed,Pkg
addprocs(3)

# Import packages
@everywhere using Flux, DiffEqSensitivity, DifferentialEquations
@everywhere using Distributions, Distances,Random
@everywhere using DelimitedFiles, Plots

@everywhere include("../utils.jl")

# Define time delay and truncation
@everywhere τ = 10
@everywhere N = 120

# Define kinetic parameters
@everywhere ps_matrix_bursty = readdlm("Fig5bcd_Fig6bv2/train_data/ps_bursty.txt")
@everywhere batchsize_bursty = size(ps_matrix_bursty,2)
@everywhere a_list = ps_matrix_bursty[1,:]
@everywhere b_list = ps_matrix_bursty[2,:]

# Load training data 
train_sol1 = readdlm("Fig5bcd_Fig6bv2/train_data/matrix_bursty_10-10.txt") # Attrtibute = 0   τ ~ Uniform(10,10)
train_sol2 = readdlm("Fig5bcd_Fig6bv2/train_data/matrix_bursty_0-20.txt")  # Attrtibute = 1   τ ~ Uniform(0,20)

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
@everywhere function f1!(x,p1,p2,ϵ,a,b)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,0)
    NN = re2_1(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

@everywhere function f2!(x,p1,p2,ϵ,a,b)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,1)
    NN = re2_2(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

# Define the CME solver
@everywhere sol_bursty1(p1,p2,ϵ,a,b,P0) = nlsolve(x->f1!(x,p1,p2,ϵ,a,b),P0).zero
@everywhere sol_bursty2(p1,p2,ϵ,a,b,P0) = nlsolve(x->f2!(x,p1,p2,ϵ,a,b),P0).zero

@everywhere function solve_bursty1(a,b,p1,p2,ϵ)
    P_0_distribution = NegativeBinomial(a*τ, 1/(1+b))
    P_0 = [pdf(P_0_distribution,i) for i=0:N-1]
    solution = sol_bursty1(p1,p2,ϵ,a,b,P_0)
    return solution
end

@everywhere function solve_bursty2(a,b,p1,p2,ϵ)
    P_0_distribution = NegativeBinomial(a*τ, 1/(1+b))
    P_0 = [pdf(P_0_distribution,i) for i=0:N-1]
    solution = sol_bursty2(p1,p2,ϵ,a,b,P_0)
    return solution
end

# Define loss function
@everywhere function loss_func1(p1,p2,ϵ)
    sol_cme = hcat(pmap(i->solve_bursty1(a_list[i],b_list[i],p1,p2,ϵ),1:batchsize_bursty)...);
    mse = Flux.mse(sol_cme,train_sol1)
    print("mse:",mse,"\n")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[:,i]), latent_size) for i=1:batchsize_bursty]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:batchsize_bursty])/batchsize_bursty
    print("kl:",kl,"\n")

    loss = λ*mse + kl
    print("loss:",loss,"\n")
    return loss
end

@everywhere function loss_func2(p1,p2,ϵ)
    sol_cme = hcat(pmap(i->solve_bursty2(a_list[i],b_list[i],p1,p2,ϵ),1:batchsize_bursty)...);
    mse = Flux.mse(sol_cme,train_sol2)
    print("mse:",mse,"\n")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[:,i]), latent_size) for i=1:batchsize_bursty]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:batchsize_bursty])/batchsize_bursty
    print("kl:",kl,"\n")

    loss = λ*mse + kl
    print("loss:",loss,"\n")
    return loss
end

function loss_func(p1,p2,ϵ)
    loss = loss_func1(p1,p2,ϵ) + loss_func2(p1,p2,ϵ)
    return loss
end

λ = 1e6
@time loss_bursty = loss_func(params1,params2,ϵ)
@time grads = gradient(()->loss_func(params1,params2,ϵ) , ps)

# Training process
lr_list = [0.02,0.01,0.008,0.006,0.004,0.002,0.001]
for lr in lr_list
    opt= ADAM(lr);
    epochs = 30
    print("learning rate = ",lr,"\n")

    @time for epoch in 1:epochs
        print(epoch,"\n")
        ϵ = rand(Normal(),latent_size)
        grads = gradient(()->loss_func(params1,params2,ϵ) , ps);
        Flux.update!(opt, ps, grads)
    end
end

# Write trained VAE parameters
using CSV,DataFrames
df = DataFrame(params1 = vcat(params1,[0 for i=1:length(params2)-length(params1)]),params2 = params2)
CSV.write("Fig5bcd_Fig6bv2/params_trained.csv",df)

# Read trained VAE parameters
using CSV,DataFrames
df = CSV.read("Fig5bcd_Fig6bv2/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]

# Solve the CME
ϵ = zeros(latent_size)
@time solution_bursty1 = hcat(pmap(i->solve_bursty1(a_list[i],b_list[i],params1,params2,ϵ),1:batchsize_bursty)...);
@time solution_bursty2 = hcat(pmap(i->solve_bursty2(a_list[i],b_list[i],params1,params2,ϵ),1:batchsize_bursty)...);
mse_bursty1 = Flux.mse(solution_bursty1,train_sol1)
mse_bursty2 = Flux.mse(solution_bursty2,train_sol2)

# Plot probability distribution
function plot_distribution(set)
    plot(0:N-1,solution_bursty2[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol2[:,set],linewidth = 3,label="Exact",title=join([round.(ps_matrix_bursty[:,set],digits=4)]),line=:dash)
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

for i = 1:5
    p = plot_channel(i)
    savefig(p,"Fig5b/results/train_results/fig_Attri=1_$i.svg")
end
