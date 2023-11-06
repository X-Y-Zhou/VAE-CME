using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots,CSV,DataFrames
using BlackBoxOptim

include("C:/Users/16037/Documents/GitHub/VAE-CME/utils.jl")

a = 0.0282
b = 3.46
N = 65
τ = 120

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
df = CSV.read("C:/Users/16037/Documents/GitHub/VAE-CME/Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_logn/params_tfo2.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length_2]

P_0_distribution_Extenicity = NegativeBinomial(a*τ, 1/(1+b));
P_0_Extenicity = [pdf(P_0_distribution_Extenicity,i) for i=0:N-1]
sol_Extenicity(p1,p2,a,b,Attribute,ϵ,P_0_Extenicity) = nlsolve(x->f_Extenicity!(x,p1,p2,a,b,Attribute,ϵ),P_0_Extenicity).zero

Attribute = 1
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
set = 13
width = "1.0-sqrt(6.0)"
# width = "2.0-sqrt(4.0)"

SSA_data = readdlm("C:/Users/16037/Documents/GitHub/VAE-CME/Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_logn/data/set$set/$(width).csv",',')[2:end,:]

@time for dataset = 1:5
print(dataset,"\n")
# SSA_data = readdlm("C:/Users/16037/Documents/GitHub/VAE-CME/Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_logn/Inference_data/set$set/$(width)_$dataset.csv",',')[2:end,:]
SSA_data = readdlm("C:/Users/16037/Documents/GitHub/VAE-CME/Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_logn/data/set$set/$(width).csv",',')[2:end,:]

# kinetic_params0 = [0.0232,2.96,0.7962]
SRange = [(0,0.06),(0,6),(0,1)]
res = bboptimize(Objective_func; Method = :adaptive_de_rand_1_bin_radiuslimited, 
SearchRange = SRange, NumDimensions = 3, MaxSteps = 300) #参数推断求解
thetax = best_candidate(res) #优化器求解参数

α = thetax[1]
β = thetax[2]
Attribute = thetax[3]

μ = 3*(1-Attribute)
σ = sqrt(2*(4-μ))
distribution = LogNormal(μ,σ)
variance = var(distribution)
bestfitness = best_fitness(res)

push!(result_list,[α,β,Attribute,μ,σ,variance,bestfitness])
end

result_list[1]
result_list[2]
result_list[3]
result_list[4]
result_list[5]

using DataFrames,CSV
df = DataFrame(result_list,:auto)
CSV.write("C:/Users/16037/Documents/GitHub/VAE-CME/Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_logn/infer_set$(set)_$(width)_exact.csv",df)

