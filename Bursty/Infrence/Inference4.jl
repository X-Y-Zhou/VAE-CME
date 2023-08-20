using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots,CSV,DataFrames
using BlackBoxOptim

include("../../utils.jl")

function bursty(N,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

N = 64

a = 0.0282
b = 3.46
τ = 120

# # Exact data
# end_time = 600
# train_sol = zeros(N+1,end_time+1)
# for i = 0:end_time
#     if i < 120
#         train_sol[1:N+1,i+1] = bursty(N+1,i)
#     else
#         train_sol[1:N+1,i+1] = bursty(N+1,120)
#     end
# end

# Simulation data
SSA_data = readdlm("Bursty/Infrence/data/SSA_1.csv",',')[2:end,:]

# Model initialization
latent_size = 10;
encoder = Chain(Dense(N+1, 20,tanh),Dense(20, latent_size * 2));
decoder = Chain(Dense(latent_size, 20,tanh),Dense(20, N),x-> 0.03*x.+[i/τ  for i in 1:N],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

length_1 = length(params1)
length_2 = length(params2)

# Define the CME
function CME(du, u, p, t)

    τ = p[end]
    
    latent_size = 10;
    encoder = Chain(Dense(N+1, 20,tanh),Dense(20, latent_size * 2));
    decoder = Chain(Dense(latent_size, 20,tanh),Dense(20, N),x-> 0.03*x.+[i/τ  for i in 1:N],x ->relu.(x));
    _, re1 = Flux.destructure(encoder);
    _, re2 = Flux.destructure(decoder);

    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[length([params1;params2])+1:length([params1;params2])+latent_size])
    NN = re2(p[length(params1)+1:length([params1;params2])])(z)

    a = p[end-2]
    b = p[end-1]

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

using CSV,DataFrames
df = CSV.read("Bursty/params_ode.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

u0 = [1.; zeros(N)];
use_time=600
time_step = 1.0; 
tspan = (0.0, use_time);
params_all = [params1;params2;zeros(latent_size);a;b;τ];
problem = ODEProblem(CME, u0, tspan,params_all);
solution = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)))
solution = solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)).u

# Check probability distribution
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
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

sample_size = 1e4
solution = set_one.(solution)
log_value = [log.(solution[i]) for i=1:use_time+1]
# log_value = [log.(train_sol[:,i]) for i=1:use_time+1]

# SSA data
SSA_timepoints = []
for i = 1:use_time+1
    temp = length(counts(round.(Int, SSA_data[i,:])))
    if  temp < N+1
        push!(SSA_timepoints,[counts(round.(Int, SSA_data[i,:]));zeros(N+1-temp)])
    else
        push!(SSA_timepoints,counts(round.(Int, SSA_data[i,:]))[1:N+1])
    end
end

# # exact data 
# SSA_timepoints = [round.(Int, train_sol[:,i].*sample_size) for i=1:use_time+1]

logp_x_z = sum([sum(SSA_timepoints[i].*log_value[i])/sample_size for i=2:use_time+1])/use_time


# a b τ
kinetic_params = [0.0282,3.46,120]

function LogLikelihood(kinetic_params)
    a = kinetic_params[1]
    b = kinetic_params[2]
    τ = kinetic_params[3]

    df = CSV.read("Bursty/params_ode.csv",DataFrame)
    params1 = df.params1
    params2 = df.params2[1:length_2]

    u0 = [1.; zeros(N)];
    use_time=600;
    time_step = 1.0; 
    tspan = (0.0, use_time);
    params_all = [params1;params2;zeros(latent_size);a;b;τ];
    problem = ODEProblem(CME, u0, tspan,params_all);
    solution = solve(problem, Tsit5(), u0=u0,p=params_all, saveat=0:time_step:Int(use_time)).u

    solution = set_one.(solution)
    log_value = [log.(solution[i]) for i=1:use_time+1]
    loglikelihood_value = -sum([sum(SSA_timepoints[i].*log_value[i])/sample_size for i=2:use_time+1])/use_time

    return loglikelihood_value
end

LogLikelihood(kinetic_params)

kinetic_params0 = [0.05,3,100]
SRange = [(0.01,0.1),(0,6),(50,150)]
res = bboptimize(LogLikelihood,kinetic_params0 ; Method = :adaptive_de_rand_1_bin_radiuslimited, SearchRange = SRange, NumDimensions = 3, MaxSteps = 100) #参数推断求解
thetax = best_candidate(res) #优化器求解参数



