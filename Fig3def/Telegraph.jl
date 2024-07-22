# Import packages
using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../utils.jl")

# Load training data 
train_sol = readdlm("Fig3def/data/Telegraph.txt")

# Truncation
N = 45

# Define kinetic parameters and time delay
sigma_on=1.0;
sigma_off=1.0;
rho_on=20.0; 
rho_off=0.0;
gamma=0.0;
τ = 1

# Model initialization
latent_size = 5;
encoder = Chain(Dense(2*(N+1), 10,tanh),Dense(10, latent_size * 2));
decoder_1 = Chain(Dense(latent_size, 10,tanh),Dense(10, N),x-> x.+[i/τ for i in 1:N],x ->relu.(x));
decoder_2  = Chain(Dense(latent_size, 10,tanh),Dense(10, N),x-> x.+[i/τ for i in 1:N],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2_1, re2_1 = Flux.destructure(decoder_1);
params2_2, re2_2 = Flux.destructure(decoder_2);
ps = Flux.params(params1,params2_1,params2_2);

# Define the CME
function CME(du, u, p, t)
    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])

    NN1 = re2_1(p[length(params1)+1:length(params1)+length(params2_1)])(z)
    NN2 = re2_2(p[length(params1)+length(params2_1)+1:end-latent_size])(z)

    du[1] = (-sigma_on-rho_off)*u[1] + (-gamma+NN1[1])*u[2] + sigma_off*u[N+2]
    for i in 2:N
        du[i] = rho_off*u[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*u[i] + (-i*gamma+NN1[i])*u[i+1] + sigma_off*u[i+N+1];
    end
    du[N+1] = rho_off*u[N] + (-sigma_on-rho_off+N*gamma-NN1[N])*u[N+1] + sigma_off*u[2*N+2];

    du[N+2] = sigma_on*u[1] + (-sigma_off-rho_on)*u[N+2] + (-gamma+NN2[1])*u[N+3]
    for i in (N+3):(2*N+1)
        du[i] = sigma_on*u[i-N-1] + rho_on*u[i-1] + (-sigma_off-rho_on+(i-N-2)*gamma -NN2[i-N-2])*u[i] + (-(i-N-1)*gamma+NN2[i-N-1])*u[i+1]
    end
    du[2*N+2] = sigma_on*u[N+1] +rho_on*u[2*N+1]+(-sigma_off-rho_on+N*gamma -NN2[N])*u[2*N+2]
end

# Read trained VAE parameters
using CSV,DataFrames
df = CSV.read("Fig3def/params_trained/Telegraph.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2_1 = df.params2_1[1:length(params2_1)]
params2_2 = df.params2_2[1:length(params2_2)]

# Solve the ODE
u0 = [1.; zeros(2*N+1)]
tf = 10;
tspan = (0, tf);
saveat = 0:0.1:10
ϵ = zeros(latent_size)
params_all = [params1;params2_1;params2_2;ϵ];
problem = ODEProblem(CME, u0, tspan, params_all);
solution = Array(solve(problem,Tsit5(),u0=u0,p=params_all,saveat=saveat))
solution = (solution[1:N+1, :] + solution[N+2:end, :]);
Flux.mse(solution,train_sol)

# Plot probability distribution
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label="SSA",title=join(["t=",time_choose/10]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(5)
    p2 = plot_distribution(10)
    p3 = plot_distribution(15)
    p4 = plot_distribution(20)
    p5 = plot_distribution(30)
    p6 = plot_distribution(40)
    p7 = plot_distribution(50)
    p8 = plot_distribution(60)
    p9 = plot_distribution(70)
    p10 = plot_distribution(80)
    p11 = plot_distribution(90)
    p12 = plot_distribution(100)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
# savefig("Fig3def/results/Telegraph.svg")



