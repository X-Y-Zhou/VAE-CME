# Import packages
using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../utils.jl")

# Load training data 
train_sol = readdlm("Fig3def/data/Bursty.txt")

# Truncation
N = 64

# Define kinetic parameters and time delay
a = 0.0282
b = 3.46
τ = 120

# Model initialization
latent_size = 10;
encoder = Chain(Dense(N+1, 20,tanh),Dense(20, latent_size * 2));
decoder = Chain(Dense(latent_size, 20,tanh),Dense(20, N),x-> 0.03*x.+[i/120  for i in 1:N],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

# Define the CME
function CME(du, u, p, t)
    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
    NN = re2(p[length(params1)+1:end-latent_size])(z)

    du[1] = (-a*b/(1+b))*u[1] + NN[1]*u[2];
    for i in 2:N
            du[i] =  (-a*b/(1+b) - NN[i-1])*u[i] + NN[i]*u[i+1];
            for j in 1:i-1
                    du[i] += (a*(b/(1+b))^j/(1+b)) * u[i-j]
            end
    end
    du[N+1] = (-a*b/(1+b) - NN[N])*u[N+1];
    for j in 1:N
            du[N+1] += (a*(b/(1+b))^j/(1+b)) * u[N+1-j]
    end
end

# Initialize the ODE solver
u0 = [1.; zeros(N)]
tf = 1200
tspan = (0, tf)
saveat = 10:10:tf
ϵ = zeros(latent_size)
params_all = [params1;params2;ϵ];
problem = ODEProblem(CME, u0, tspan, params_all);
solution = solve(problem,Tsit5(),u0=u0,p=params_all,saveat=saveat)

# Define loss function
function loss_func(p1,p2,ϵ)
    params_all = [p1;p2;ϵ]
    sol_cme = solve(ODEProblem{true, SciMLBase.FullSpecialize}(CME, u0, tspan, params_all),Tsit5(),u0=u0,p=params_all,saveat=saveat)
    temp = sol_cme.u

    mse = Flux.mse(Array(sol_cme),train_sol[:,saveat.+1])
    print("mse:",mse," ")

    μ_logσ_list = [split_encoder_result(re1(p1)(temp[i]), latent_size) for i=1:length(saveat)]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:length(saveat)])/length(saveat)
    print(kl," ")

    loss = λ*mse + kl
    print(loss,"\n")
    return loss
end

λ = 1e5
ϵ = zeros(latent_size)
loss_func(params1,params2,ϵ)
@time grads = gradient(()->loss_func(params1,params2,ϵ),ps)

# Training process
lr_list = [0.02,0.01,0.008,0.006,0.004]
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
df = DataFrame( params1 = vcat(params1,[0 for i=1:length(params2)-length(params1)]),params2 = params2)
CSV.write("Fig3def/params_trained/Birth_Death.csv",df)

# Read trained VAE parameters
using CSV,DataFrames
df = CSV.read("Fig3def/params_trained/Bursty.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]

# Solve the ODE
u0 = [1.; zeros(N)]
tf = 1200
tspan = (0, tf)
saveat = 0:1:tf
ϵ = zeros(latent_size)
params_all = [params1;params2;ϵ];
problem = ODEProblem(CME, u0, tspan, params_all);
solution = Array(solve(problem,Tsit5(),u0=u0,p=params_all,saveat=saveat))

# Plot probability distribution
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label="SSA",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(27)
    p2 = plot_distribution(47)
    p3 = plot_distribution(67)
    p4 = plot_distribution(77)
    p5 = plot_distribution(87)
    p6 = plot_distribution(97)
    p7 = plot_distribution(107)
    p8 = plot_distribution(120)
    p9 = plot_distribution(200)
    p10 = plot_distribution(300)
    p11 = plot_distribution(500)
    p12 = plot_distribution(800)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
# savefig("Fig3def/results/Bursty.svg")
