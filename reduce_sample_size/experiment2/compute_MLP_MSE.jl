using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

solnet_X = readdlm("reduce_sample_size/experiment2/data/X_10000.csv",',')[2:end,:]
solnet_Y = readdlm("reduce_sample_size/experiment2/data/Y_10000.csv",',')[2:end,:]

maximum(solnet_X)
maximum(solnet_Y)

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
train_sol_X

train_sol_Y = zeros(N+1,size(solnet_Y,1))
for i =1:size(solnet_Y,1)
    probability = convert_histo(vec(solnet_Y[i,:]))[2]
    if length(probability)<N+1
        train_sol_Y[1:length(probability),i] = probability
    else
        train_sol_Y[1:N+1,i] = probability[1:N+1]
    end
end
train_sol_Y

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

model = Chain(Dense((N+1)*(N+1), 5, tanh), Dense(5, N*(N-1)), x -> x .+ bias, x->relu.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);
p1

function CME(du, u, p, t)
    NN = re(p)(u)

    Na(k) = diagm(0=>[0.0; NN[(N-1)*k+1:(N-1)*k+N-1]; 0.0])
    Nb(k) = diagm(1=>[NN[(N-1)*k+1:(N-1)*k+N-1]; 0.0])
    # m = 1
        du[1:N+1] = (D1(0)-D2(0)-Na(0)) * u[1:(N+1)] +
                                 D2(1) * u[(N+1+1):(N+1+N+1)]
    # m=2->N
        for m in 2:N
               du[((N+1)*(m-1)+1):((N+1)*(m-1)+N+1)] = Nb(m-2) * u[((N+1)*(m-2)+1):((N+1)*(m-2)+N+1)] +
                                                                                                   (D1(m-1)-D2(m-1)-Na(m-1)) * u[((N+1)*(m-1)+1):((N+1)*(m-1)+N+1)] +
                                                                                                   D2(m) * u[((N+1)*m+1):((N+1)*m+N+1)]
        end
   # m =N+1
        du[((N+1)*N+1):((N+1)*N+N+1)] = (Nb(N-1)) * u[(N+1)*(N-1)+1:(N+1)*(N-1)+N+1] +
                                                                             (D1(N)-D2(N))*u[(N+1)*N+1:(N+1)*N+N+1]
end

using CSV,DataFrames
Sample_size = 10000
df = CSV.read("reduce_sample_size/experiment2/params_MLP_$Sample_size.csv",DataFrame)
p1 = df.p
ps = Flux.params(p1);

#check
u0 = [1.; zeros((N+1)*(N+1)-1)]
tf = 200.0
tspan = (0.0, tf)
time_step = 1.0
problem = ODEProblem(CME, u0, tspan, p1)
solution = solve(problem, Tsit5(), u0=u0, p=p1,saveat=0:time_step:Int(tf))

tmax = Int(tf+1)
sol_X = zeros(N+1,tmax)
sol_Y = zeros(N+1,tmax)
for i = 1:tmax
    sol_X[:,i]=sum(reshape(solution[:,i],N+1,N+1),dims=2)[1:N+1]
    sol_Y[:,i]=sum(reshape(solution[:,i],N+1,N+1),dims=1)[1:N+1]
end

Flux.mse(sol_X,train_sol_X)
Flux.mse(sol_Y,train_sol_Y)