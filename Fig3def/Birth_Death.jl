# Import packages
using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../utils.jl")

# Load training data 
train_sol = readdlm("Fig2b/data/Birth_death.txt")

# Truncation
N = 271

# Define kinetic parameters and time delay
τ = 10
ρ = 20

# Model initialization
latent_size = 10;
encoder = Chain(Dense(N+1, 10,tanh),Dense(10, latent_size * 2));
decoder = Chain(Dense(latent_size, 10,tanh),Dense(10, N),x-> x.+[i/τ  for i in 1:N],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

# Define the CME
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

# Read trained VAE parameters
using CSV,DataFrames
df = CSV.read("Fig2b/params_trained/Birth_Death.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2

# Solve the ODE
u0 = [1.; zeros(N)];
tf = 100;
tspan = (0, tf)
saveat = 0:1:tf
params_all = [params1;params2;zeros(latent_size)];
problem = ODEProblem(CME, u0, tspan,params_all);
solution = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=saveat))

# Plot probability distribution
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label="SSA",title=join(["t=",time_choose]),line=:dash)
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
# savefig("Fig2b/results/Birth_Death.svg")


