using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../utils.jl")

data = readdlm("Birth-Death/data.csv", ',')[2:end,:]

N = Int(maximum(data))

# Load train data
train_sol = zeros(N+1,size(data,1))
for i =1:size(data,1)
    probability = convert_histo(vec(data[i,:]))[2]
    if length(probability)<N+1
        train_sol[1:length(probability),i] = probability
    else
        train_sol[1:N+1,i] = probability[1:N+1]
    end
end
train_sol

# Exact solution
function birth_death(N,τ)
    distribution = Poisson(ρ*τ)
    P = zeros(N)
    for i=1:N
        P[i] = pdf(distribution,i)
    end
    return P
end;

τ = 10
ρ = 20

end_time = 100
exact_sol = zeros(N+1,end_time+1)
for i = 0:end_time
    if i < τ
        exact_sol[1:N+1,i+1] = birth_death(N+1,i)
    else
        exact_sol[1:N+1,i+1] = birth_death(N+1,τ)
    end
end

# model initialization
latent_size = 10;
encoder = Chain(Dense(N+1, 10,tanh),Dense(10, latent_size * 2));
decoder = Chain(Dense(latent_size, 10,tanh),Dense(10, N),x-> x.+[i/τ  for i in 1:N],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

function CME(du, u, p, t)
    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
    NN = re2(p[length(params1)+1:end-latent_size])(z)

    du[1] = -ρ*u[1] + NN[1]*u[2];
    for i in 2:N
          du[i] = ρ*u[i-1] + (-ρ-NN[i-1])*u[i] + NN[i]*u[i+1];
    end
    du[N+1] = ρ*u[N] + (-ρ-NN[N])*u[N+1]
end

# initialize the ODE solver
u0 = [1.; zeros(N)]
tf = 100; #end time
tspan = (0, tf);
saveat = 0:1:100
ϵ = zeros(latent_size)
params_all = [params1;params2;ϵ]
problem = ODEProblem(CME, u0, tspan, params_all);
solution = Array(solve(problem,Tsit5(),u0=u0,p=params_all,saveat=saveat))

function loss_func(p1,p2,ϵ)
    params_all = [p1;p2;ϵ]
    sol_cme = solve(ODEProblem(CME, u0, tspan, params_all),Tsit5(),u0=u0,p=params_all,saveat=saveat)
    
    temp = sol_cme.u

    μ_logσ_list = [split_encoder_result(re1(p1)(temp[i]), latent_size) for i=1:length(saveat)]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:length(saveat)])/length(saveat)
    print("kl:",kl,"\n")

    reg_zero = Flux.mse(Array(sol_cme),train_sol[:,saveat.+1])
    print("mse:",reg_zero," ")

    loss = kl + λ2*reg_zero

    print(loss,"\n")
    return loss
end

λ2 = 50000000

ϵ = zeros(latent_size)
loss_func(params1,params2,ϵ)
grads = gradient(()->loss_func(params1,params2,ϵ),ps)

epochs_all = 0

lr = 0.02;
opt= ADAM(lr);
epochs = 20;
epochs_all = epochs_all + epochs
print("learning rate = ",lr)

mse_list = []

@time for epoch in 1:epochs
    ϵ = rand(Normal(),latent_size)
    print(epoch,"\n")
    grads = gradient(()->loss_func(params1,params2,ϵ) , ps)
    Flux.update!(opt, ps, grads)
end

# write parameters
using CSV,DataFrames
df = DataFrame( params1 = vcat(params1,[0 for i=1:length(params2)-length(params1)]),params2 = params2)
CSV.write("Research/machine-learning/ode/birth-death/params_trained.csv",df)

# check
using CSV,DataFrames
df = CSV.read("Birth-Death/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

u0 = [1.; zeros(N)];
use_time=100;
time_step = 1.0; 
tspan = (0.0, use_time);
params_all = [params1;params2;zeros(latent_size)];
problem = ODEProblem(CME, u0, tspan,params_all);
solution = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)))

Flux.mse(solution,train_sol)

mean_exact = [sum([j for j=0:N].*exact_sol[:,i]) for i=1:size(exact_sol,2)]
mean_trained = [sum([j for j=0:N].*solution[:,i]) for i=1:size(solution,2)]
plot(mean_trained,linewidth = 3,label="VAE-CME",xlabel = "# t", ylabel = "mean value")
plot!(mean_exact,label="exact",linewidth = 3,line=:dash,legend=:bottomright)

#check probabilities
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,exact_sol[:,time_choose+1],linewidth = 3,label="Exact",title=join(["t=",time_choose]),line=:dash,legend=:bottomleft)
    return p
end

function plot_all()
    p1 = plot_distribution(2)
    p2 = plot_distribution(4)
    p3 = plot_distribution(8)
    p4 = plot_distribution(16)
    p5 = plot_distribution(20)
    p6 = plot_distribution(40)
    p7 = plot_distribution(60)
    p8 = plot_distribution(80)
    p9 = plot_distribution(100)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,size=(1200,900))
end
plot_all()
savefig("Birth-Death/fitting.svg")