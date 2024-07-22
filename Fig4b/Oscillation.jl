# Import packages
using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../utils.jl")

# Load check data
check_sol_X = readdlm("Fig4b/data/X_check.txt")
check_sol_Y = readdlm("Fig4b/data/Y_check.txt")

# Truncation
N = 26

# Define kinetic parameters and time delay
k_1S = 1.
k_d = 1.
k_p = 2.
k_2E_T = 1.
Km = 1.
τ = 10

J1(Y) = k_1S * k_d^k_p / (k_d^k_p + Y^k_p)
J2(Y) = k_2E_T / (Km + Y)
D1(m) = diagm(-1=>fill(J1(m), N)) .+ diagm(0=>[fill(-J1(m), N);0.0])
D2(m) = m*J2(m) * diagm(0=>fill(1.0, N+1))

# Define bias
bias = zeros(N*(N-1))
for i in 0:N-1
        bias[1+i*(N-1):N-1+i*(N-1)] = [i/τ for i in 1:N-1]
end

# Model initialization
latent_size = 10;
encoder = Chain(Dense((N+1)*(N+1), 5,tanh),Dense(5, latent_size * 2));
decoder = Chain(Dense(latent_size, 5,tanh),Dense(5, N*(N-1)), x -> x .+ bias, x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

# Define the CME
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

# Read trained VAE parameters
using CSV,DataFrames
df = CSV.read("Fig4b/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]

# Solve the ODE
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
Flux.mse(sol_X,check_sol_X)
Flux.mse(sol_Y,check_sol_Y)

# Plot probability distribution
function plot_distribution_X(time_choose)
    p=plot(0:N,sol_X[:,time_choose+1],label="X",linewidth = 3,xlabel = "# of products", ylabel = "Probability")
    plot!(0:N,check_sol_X[:,time_choose+1],linewidth = 3,label="SSA",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_distribution_Y(time_choose)
    p=plot(0:N,sol_Y[:,time_choose+1],label="Y",linewidth = 3,xlabel = "# of products", ylabel = "Probability")
    plot!(0:N,check_sol_Y[:,time_choose+1],linewidth = 3,label="SSA",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_distribution_X_all()
    p1 = plot_distribution_X(28)
    p2 = plot_distribution_X(27)
    p3 = plot_distribution_X(30)
    p4 = plot_distribution_X(40)
    p5 = plot_distribution_X(50)
    p6 = plot_distribution_X(75)
    p7 = plot_distribution_X(100)
    p8 = plot_distribution_X(150)
    p9 = plot_distribution_X(200)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,size=(1200,800))
end
plot_distribution_X_all()
# savefig("Fig4b/results/X.svg")

function plot_distribution_Y_all()
    p1 = plot_distribution_Y(28)
    p2 = plot_distribution_Y(27)
    p3 = plot_distribution_Y(30)
    p4 = plot_distribution_Y(40)
    p5 = plot_distribution_Y(50)
    p6 = plot_distribution_Y(75)
    p7 = plot_distribution_Y(100)
    p8 = plot_distribution_Y(150)
    p9 = plot_distribution_Y(200)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,size=(1200,800))
end
plot_distribution_Y_all()
# savefig("Fig4b/results/Y.svg")







