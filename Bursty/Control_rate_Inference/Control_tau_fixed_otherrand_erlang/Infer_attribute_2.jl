using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots,CSV,DataFrames
using BlackBoxOptim

include("../../../utils.jl")

a = 0.0282
b = 3.46
N = 81
τ = 120

# Simulation data
# SSA_data = readdlm("Control_rate_Inference/Control_tau_fixed_otherrand_erlang/Inference_data/set1/30-210_1.csv",',')[2:end,:]

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
df = CSV.read("Control_rate_Inference/Control_tau_fixed_otherrand_erlang/params_ct2.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

P_0_distribution_Extenicity = NegativeBinomial(a*τ, 1/(1+b));
P_0_Extenicity = [pdf(P_0_distribution_Extenicity,i) for i=0:N-1]
sol_Extenicity(p1,p2,a,b,Attribute,ϵ,P_0_Extenicity) = nlsolve(x->f_Extenicity!(x,p1,p2,a,b,Attribute,ϵ),P_0_Extenicity).zero

Attribute = 0
ϵ = zeros(latent_size)
solution = sol_Extenicity(params1,params2,a,b,Attribute,ϵ,P_0_Extenicity)

sample_size = 5000
solution = set_one(solution)
log_value = log.(solution)

# SSA data
result_list = []
set = 1
width = "5-24"

for dataset = 1:5
print(dataset,"\n")
SSA_data = readdlm("Control_rate_Inference/Control_tau_fixed_otherrand_erlang/Inference_data/set$set/$(width)_$dataset.csv",',')[2:end,:]
SSA_timepoints = round.(Int, vec(SSA_data).*sample_size)

# Ex = P2mean(SSA_data)
# Dx = P2var(SSA_data)

# a = 2Ex^2/(Dx-Ex)τ
# b = (Dx-Ex)/2Ex

function LogLikelihood(kinetic_params)
    a = 0.0282
    b = 3.46
    Attribute = kinetic_params[1]

    df = CSV.read("Control_rate_Inference/Control_tau_fixed_otherrand_erlang/params_ct2.csv",DataFrame)
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

kinetic_params0 = [0.5]
SRange = [(0,1)]
res = bboptimize(LogLikelihood,kinetic_params0 ; Method = :adaptive_de_rand_1_bin_radiuslimited, 
SearchRange = SRange, NumDimensions = 1, MaxSteps = 150) #参数推断求解
thetax = best_candidate(res) #优化器求解参数
# best_fitness(res)

α = 0.0282
β = 3.46
Attribute = thetax[1]
x1 = Int(round(exp((1-Attribute)*log(30)),digits=0))
x2 = τ/x1
distribution = Erlang(x1,x2)
variance = var(distribution)

push!(result_list,[α,β,Attribute,x1,x2,variance])
end

result_list
result_list[1]
result_list[2]
result_list[3]
result_list[4]
result_list[5]

using DataFrames,CSV
df = DataFrame(result_list,:auto)
CSV.write("Control_rate_Inference/Control_tau_fixed_otherrand_erlang/temp_2.csv",df)

function check_inference(kinetic_params)
    a = kinetic_params[1]
    b = kinetic_params[2]
    Attribute = kinetic_params[3]
    # Attribute = 0.75

    df = CSV.read("Control_rate_Inference/Control_tau_fixed_otherrand_erlang/params_ct2.csv",DataFrame)
    params1 = df.params1
    params2 = df.params2[1:length_2]

    ϵ = zeros(latent_size)
    P_0_distribution_Extenicity = NegativeBinomial(a*τ, 1/(1+b));
    P_0_Extenicity = [pdf(P_0_distribution_Extenicity,i) for i=0:N-1]
    solution = sol_Extenicity(params1,params2,a,b,Attribute,ϵ,P_0_Extenicity)
    return solution
end

set
width
result_list

dataset = 1
result_list[dataset]
solution_inference = check_inference(result_list[dataset])
solution_theoty = check_inference([0.0232,2.96,0.75])
SSA_data = vec(readdlm("Control_rate_Inference/Control_tau_fixed_otherrand_erlang/data/set$set/$(width).csv",',')[2:end,:])

using Flux
Flux.mse(solution_inference,SSA_data)
Flux.mse(solution_theoty,SSA_data)

plot(0:N-1,solution_inference,lw=3,label="inference")
plot!(0:N-1,solution_theoty,lw=3,label="theory",line=:dash)
plot!(0:N-1,SSA_data,lw=3,label="SSA",line=:dash)

Attribute = 0.1
x1 = exp((1-Attribute)*log(30))
x2 = τ/x1

Int(x1)
Int(round(x1,digits=0))


x_temp = [1,2,5,10,20,30]
x_temp_log = log.(x_temp)
y_temp = [1,0.8,0.4,0.2,0.1,0]
y_temp_theory = (-1/log(30)).*(log.(x_temp)).+1

# Erlang(a,b)
# a = 30;b = 4   # var = 480   [0]    [0]
# a = 20;b = 6   # var = 720   [0.1]  [0.1192]
# a = 10;b = 12  # var = 1440  [0.2]  [0.3230]
# a = 5; b = 24  # var = 2880  [0.4]  [0.5268]
# a = 2; b = 60  # var = 7200  [0.8]  [0.7962]
# a = 1; b = 120 # var = 14400 [1]    [1]

Attribute_list1 = [1,0.715,0.412,0.128,0]
Attribute_list2 = [1,0.75,0.5,0.25,0]
τ_list = [0,30,60,90,120]
plot(τ_list,Attribute_list1,label="inference")
plot!(τ_list,Attribute_list2,label="theory")
scatter!(τ_list,Attribute_list1,label=:false)
scatter!(τ_list,Attribute_list2,label=:false)
