using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots,CSV,DataFrames
using BlackBoxOptim

include("../../../utils.jl")

a = 0.0282
b = 3.46
N = 65
τ = 120

# Simulation data
# SSA_data = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/Inference_data/set1/30-210_1.csv",',')[2:end,:]

# model initialization
latent_size = 10;
encoder = Chain(Dense(N, 20,tanh),Dense(20, latent_size * 2));
decoder_1 = Chain(Dense(latent_size+1, 20),Dense(20 , N-1),x->0.03.* x.+[i/τ  for i in 1:N-1],x ->relu.(x));
decoder_2  = Chain(decoder_1[1],decoder_1[2],x->0.03.* x.+[i/τ  for i in 1:N-1],decoder_1[4]);

params1, re1 = Flux.destructure(encoder);
params2, re2_1 = Flux.destructure(decoder_1);
      _, re2_2 = Flux.destructure(decoder_2);
ps = Flux.params(params1,params2);

length_1 = length(params1)
length_2 = length(params2)

function f_Extenicity!(x,p1,p2,a,b,Attribute,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,Attribute)
    NN = re2_2(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

using CSV,DataFrames
df = CSV.read("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/params_tfo.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

P_0_distribution_Extenicity = NegativeBinomial(a*τ, 1/(1+b));
P_0_Extenicity = [pdf(P_0_distribution_Extenicity,i) for i=0:N-1]
sol_Extenicity(p1,p2,a,b,Attribute,ϵ,P_0_Extenicity) = nlsolve(x->f_Extenicity!(x,p1,p2,a,b,Attribute,ϵ),P_0_Extenicity).zero

Attribute = 0
ϵ = zeros(latent_size)
solution = sol_Extenicity(params1,params2,a,b,Attribute,ϵ,P_0_Extenicity)

sample_size = 1e4
solution = set_one(solution)
log_value = log.(solution)

# SSA data
result_list = []
for dataset = 1:5
SSA_data = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/Inference_data/set1/30-210_$dataset.csv",',')[2:end,:]

# SSA_data[:,1:5]
i = 1
SSA_timepoints = round.(Int, SSA_data[:,i].*sample_size)
logp_x_z = sum(SSA_timepoints.*log_value)/sample_size

# a b Attribute
# kinetic_params = [a,b,Attribute]

function LogLikelihood(kinetic_params)
    a = kinetic_params[1]
    b = kinetic_params[2]
    Attribute = kinetic_params[3]

    df = CSV.read("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/params_tfo.csv",DataFrame)
    params1 = df.params1
    params2 = df.params2[1:length_2]

    ϵ = zeros(latent_size)
    P_0_distribution_Extenicity = NegativeBinomial(a*τ, 1/(1+b));
    P_0_Extenicity = [pdf(P_0_distribution_Extenicity,i) for i=0:N-1]
    solution = sol_Extenicity(params1,params2,a,b,Attribute,ϵ,P_0_Extenicity)

    # loglikelihood_value = Flux.mse(solution,SSA_data[:,i])

    solution = set_one(solution)
    log_value = log.(solution)
    loglikelihood_value = -sum(SSA_timepoints.*log_value)/sample_size

    return loglikelihood_value
end

# LogLikelihood(kinetic_params0)

kinetic_params0 = [0.03,3,0.5]
SRange = [(0,0.06),(0,6),(0,1)]
res = bboptimize(LogLikelihood,kinetic_params0 ; Method = :adaptive_de_rand_1_bin_radiuslimited, 
SearchRange = SRange, NumDimensions = 3, MaxSteps = 100) #参数推断求解
thetax = best_candidate(res) #优化器求解参数
# best_fitness(res)

α = thetax[1]
β = thetax[2]
Attribute = thetax[3]
τ1 = (1-Attribute)*τ
τ2 = 2τ-τ1
distribution = Uniform(τ1,τ2)
var = (τ1-τ2)^2/12

[α,β,Attribute,τ1,τ2,var]
push!(result_list,[α,β,Attribute,τ1,τ2,var])
end
result_list
result_list[1]
result_list[2]
result_list[3]
result_list[4]
result_list[5]

using DataFrames,CSV
df = DataFrame(result_list,:auto)
CSV.write("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/temp_1.csv",df)


x = 1
i

# training data
# set1
# mean = 120
# α = 0.0282 β = 3.46 
# Uniform(0,240)   var = 4800  [1]
# Uniform(30,210)  var = 2700
# Uniform(60,180)  var = 1200
# Uniform(90,150)  var = 300
# Uniform(120,120) var = 0     [0]

# set2
# mean = 120
# α = 0.0082 β = 1.46 
# Uniform(0,240)   var = 4800  [1]
# Uniform(30,210)  var = 2700
# Uniform(60,180)  var = 1200
# Uniform(90,150)  var = 300
# Uniform(120,120) var = 0     [0]

# check data
# set3
# mean = 120
# α = 0.0182 β = 2.46 
# Uniform(0,240)   var = 4800 
# Uniform(30,210)  var = 2700
# Uniform(60,180)  var = 1200
# Uniform(90,150)  var = 300
# Uniform(120,120) var = 0 

# set4
# mean = 120
# α = 0.0232 β = 2.96 
# Uniform(0,240)   var = 4800 
# Uniform(30,210)  var = 2700
# Uniform(60,180)  var = 1200
# Uniform(90,150)  var = 300
# Uniform(120,120) var = 0 


