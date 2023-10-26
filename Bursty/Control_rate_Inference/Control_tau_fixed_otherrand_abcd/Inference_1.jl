using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots,CSV,DataFrames
using BlackBoxOptim

include("../../../utils.jl")

a = 0.0282
b = 3.46
N = 65
τ = 120

function f_NN(x,l,m,n,o)
    return l*x^m/(n+x^o)
end;

# model initialization
latent_size = 2;
encoder = Chain(Dense(N, 200,tanh),Dense(200, latent_size * 2));
decoder_1 = Chain(Dense(latent_size+1, 200),Dense(200, 4), x->exp.(x));
decoder_2  = Chain(decoder_1[1],decoder_1[2], x->exp.(x));

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
    l,m,n,o = re2_1(p2)(z)
    NN = f_NN.(1:N-1,l,m,n,o)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

using CSV,DataFrames
df = CSV.read("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_abcd/params_tfo2.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

P_0_distribution_Extenicity = NegativeBinomial(a*τ, 1/(1+b));
P_0_Extenicity = [pdf(P_0_distribution_Extenicity,i) for i=0:N-1]
sol_Extenicity(p1,p2,a,b,Attribute,ϵ,P_0_Extenicity) = nlsolve(x->f_Extenicity!(x,p1,p2,a,b,Attribute,ϵ),P_0_Extenicity).zero

Attribute = 0
ϵ = zeros(latent_size)
solution = sol_Extenicity(params1,params2,a,b,Attribute,ϵ,P_0_Extenicity)

function Objective_func(kinetic_params)
    a = kinetic_params[1]
    b = kinetic_params[2]
    Attribute = kinetic_params[3]

    ϵ = zeros(latent_size)
    P_0_distribution_Extenicity = NegativeBinomial(a*τ, 1/(1+b));
    P_0_Extenicity = [pdf(P_0_distribution_Extenicity,i) for i=0:N-1]
    solution = sol_Extenicity(params1,params2,a,b,Attribute,ϵ,P_0_Extenicity)
    Objective_value = Flux.mse(solution,SSA_data)

    return Objective_value
end

# SSA data
result_list = []
set = 5
width = "30-210"
width = "60-180"
width = "90-150"

dataset = 2
SSA_data = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_abcd/Inference_data/set$set/$(width)_$dataset.csv",',')[2:end,:]

@time for dataset = 1:5
print(dataset,"\n")
SSA_data = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_abcd/Inference_data/set$set/$(width)_$dataset.csv",',')[2:end,:]
# SSA_data = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_abcd/Inference_data/set$set/$(width).csv",',')[2:end,:]

# Ex = P2mean(SSA_data)
# Dx = P2var(SSA_data)

# a_0 = 2Ex^2/(Dx-Ex)τ
# b_0 = (Dx-Ex)/2Ex

# kinetic_params0 = [0.0232,2.96,0.5268]
SRange = [(0,0.06),(0,6),(0,1)]
res = bboptimize(Objective_func; Method = :adaptive_de_rand_1_bin_radiuslimited, 
SearchRange = SRange, NumDimensions = 3, MaxSteps = 300) #参数推断求解

# res = bboptimize(Objective_func,kinetic_params0; Method = :adaptive_de_rand_1_bin_radiuslimited, 
# SearchRange = SRange, NumDimensions = 3, MaxSteps = 300) #参数推断求解

thetax = best_candidate(res) #优化器求解参数

α = thetax[1]
β = thetax[2]
Attribute = thetax[3]

τ1 = (1-Attribute)*τ
τ2 = 2τ-τ1
distribution = Uniform(τ1,τ2)
variance = var(distribution)
bestfitness = best_fitness(res)

push!(result_list,[α,β,Attribute,τ1,τ2,variance,bestfitness])
end

result_list
result_list[1]
result_list[2]
result_list[3]
result_list[4]
result_list[5]

using DataFrames,CSV
df = DataFrame(result_list,:auto)
CSV.write("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_abcd/temp_1.csv",df)