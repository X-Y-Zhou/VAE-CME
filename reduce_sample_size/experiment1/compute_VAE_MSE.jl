using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

solnet_X = readdlm("reduce_sample_size/experiment1/data/X_10000.csv",',')[2:end,:]
solnet_Y = readdlm("reduce_sample_size/experiment1/data/Y_10000.csv",',')[2:end,:]

N = 26

train_sol_X = zeros(N+1,size(solnet_X,1))
for i =1:size(solnet_X,1)
    probability = convert_histo(vec(solnet_X[i,:]))[2]
    if length(probability)<N+1
        train_sol_X[1:length(probability),i] = probability
    else
        train_sol_X[1:N+1,i] = probability[1:N+1]
    end
end

train_sol_Y = zeros(N+1,size(solnet_Y,1))
for i =1:size(solnet_Y,1)
    probability = convert_histo(vec(solnet_Y[i,:]))[2]
    if length(probability)<N+1
        train_sol_Y[1:length(probability),i] = probability
    else
        train_sol_Y[1:N+1,i] = probability[1:N+1]
    end
end

k_1S=1.
k_d=1.
k_p=2.
k_2E_T=1.
Km=1.

J1(Y) = k_1S * k_d^k_p / (k_d^k_p + Y^k_p)
J2(Y) = k_2E_T / (Km + Y)
D1(m) = diagm(-1=>fill(J1(m), N)) .+ diagm(0=>[fill(-J1(m), N);0.0])
D2(m) = m*J2(m) * diagm(0=>fill(1.0, N+1))

# model initialization
bias = zeros(N*(N-1))
for i in 0:N-1
        bias[1+i*(N-1):N-1+i*(N-1)] = [i/10 for i in 1:N-1]
end

#model initialization
latent_size = 10;
encoder = Chain(Dense((N+1)*(N+1), 5,tanh),Dense(5, latent_size * 2));
decoder = Chain(Dense(latent_size, 5,tanh),Dense(5, N*(N-1)), x -> x .+ bias, x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

function CME(du, u, p, t)
    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
    NN = re2(p[length(params1)+1:end-latent_size])(z)

    Na(k) = diagm(0=>[0.0; NN[(N-1)*k+1:(N-1)*k+N-1]; 0.0])
    Nb(k) = diagm(1=>[NN[(N-1)*k+1:(N-1)*k+N-1]; 0.0])
    # m = 1
    du[1:N+1] = (D1(0)-D2(0)-Na(0)) * u[1:(N+1)] + D2(1) * u[(N+1+1):(N+1+N+1)]
    for m in 2:N
        du[((N+1)*(m-1)+1):((N+1)*(m-1)+N+1)] = Nb(m-2) * u[((N+1)*(m-2)+1):((N+1)*(m-2)+N+1)] +
                                                                                (D1(m-1)-D2(m-1)-Na(m-1)) * u[((N+1)*(m-1)+1):((N+1)*(m-1)+N+1)] +
                                                                                D2(m) * u[((N+1)*m+1):((N+1)*m+N+1)]
    end
    # m =N+1
    du[((N+1)*N+1):((N+1)*N+N+1)] = (Nb(N-1)) * u[(N+1)*(N-1)+1:(N+1)*(N-1)+N+1] + (D1(N)-D2(N))*u[(N+1)*N+1:(N+1)*N+N+1]
end

#read params
using CSV,DataFrames
Sample_size = 10000
df = CSV.read("reduce_sample_size/experiment1/params_VAE_$Sample_size.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

u0 = [1.; zeros((N+1)*(N+1)-1)]; 
tf = 200.0;
tspan = (0.0, tf);
time_step = 1.0
ϵ = zeros(latent_size)
params_all = [params1;params2;ϵ];
problem = ODEProblem(CME, u0, tspan, params_all);
solution = Array(solve(problem, Tsit5(), u0=u0, p=params_all, saveat=0:time_step:Int(tf)))

tmax = Int(tf+1)
sol_X = zeros(N+1,tmax)
sol_Y = zeros(N+1,tmax)
for i = 1:tmax
    sol_X[:,i]=sum(reshape(solution[:,i],N+1,N+1),dims=2)[1:N+1]
    sol_Y[:,i]=sum(reshape(solution[:,i],N+1,N+1),dims=1)[1:N+1]
end

# MSE
Flux.mse(sol_X,train_sol_X)
Flux.mse(sol_Y,train_sol_Y)