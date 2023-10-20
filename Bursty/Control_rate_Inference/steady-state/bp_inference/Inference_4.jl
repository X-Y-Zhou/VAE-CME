using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots,CSV,DataFrames
using BlackBoxOptim

include("../../../../utils.jl")

a = 0.0282
b = 3.46
N = 64
τ = 120

function f_NN(x,l,m,n,o)
    return l*x^m/(n+x^o)
end

# model initialization
model = Chain(Dense(N, 100, tanh), Dense(100, 4), x ->exp.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);
length_1 = length(p1)

function f1!(x,p,a,b)
    l,m,n,o = re(p)(x)
    NN = f_NN.(1:N-1,l,m,n,o)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

P_0 = [pdf(NegativeBinomial(a*τ, 1/(1+b)),j) for j=0:N-1]
sol(p,a,b,P0) = nlsolve(x->f1!(x,p,a,b),P0).zero

using CSV,DataFrames
df = CSV.read("bp_inference/params_bp_abcd4-1.csv",DataFrame)
p1 = df.p1

solution = sol(p1,a,b,P_0)

# sample_size = 5000
# solution = set_one(solution)
# log_value = log.(solution)

function Objective_func(kinetic_params)
    a = kinetic_params[1]
    b = kinetic_params[2]

    P_0 = [pdf(NegativeBinomial(a*τ, 1/(1+b)),j) for j=0:N-1]
    solution = sol(p1,a,b,P_0)
    Objective_value = Flux.mse(solution,SSA_data)

    # solution = set_one(solution)
    # log_value = log.(solution)
    # loglikelihood_value = -sum(SSA_timepoints.*log_value)/sample_size

    return Objective_value
end

# SSA data
result_list = []
set = 4
dataset = 1
SSA_data = readdlm("bp_inference/Inference_data/set$set/$dataset.csv",',')[2:end,:]

@time for dataset = 1:5
print(dataset,"\n")
SSA_data = readdlm("bp_inference/Inference_data/set$set/$dataset.csv",',')[2:end,:]
# SSA_data = readdlm("bp_inference/data/set$dataset.csv",',')[2:end,:]

# kinetic_params0 = [0.0232,2.96]
SRange = [(0,0.06),(0,6)]
res = bboptimize(Objective_func; Method = :adaptive_de_rand_1_bin_radiuslimited, 
SearchRange = SRange, NumDimensions = 2, MaxSteps = 300) #参数推断求解
thetax = best_candidate(res) #优化器求解参数

α = thetax[1]
β = thetax[2]
bestfitness = best_fitness(res)

push!(result_list,[α,β,bestfitness])
end

result_list
result_list[1]
result_list[2]
result_list[3]
result_list[4]
result_list[5]

using DataFrames,CSV
df = DataFrame(result_list,:auto)
CSV.write("bp_inference/temp_4.csv",df)

function check_inference(kinetic_params)
    a = kinetic_params[1]
    b = kinetic_params[2]
    Attribute = kinetic_params[3]

    df = CSV.read("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_erlang/params_tfo2.csv",DataFrame)
    params1 = df.params1
    params2 = df.params2[1:length_2]

    ϵ = zeros(latent_size)
    P_0_distribution_Extenicity = NegativeBinomial(a*τ, 1/(1+b));
    P_0_Extenicity = [pdf(P_0_distribution_Extenicity,i) for i=0:N-1]
    solution = sol_Extenicity(params1,params2,a,b,Attribute,ϵ,P_0_Extenicity)
    return solution
end

using Flux
set
width
result_list

dataset = 2
result_list[dataset]
solution_inference = check_inference(result_list[dataset])
SSA_data = vec(readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_erlang/data/set$set/$(width).csv",',')[2:end,:])
Flux.mse(solution_inference,SSA_data)

plot(0:N-1,solution_inference,lw=3,label="inference")
plot!(0:N-1,SSA_data,lw=3,label="SSA",line=:dash)


# Erlang(a,b)
# a = 30;b = 4   # var = 480   [0]    [0]
# a = 20;b = 6   # var = 720   [0.1]  [0.1192]
# a = 10;b = 12  # var = 1440  [0.2]  [0.3230]
# a = 5; b = 24  # var = 2880  [0.4]  [0.5268]
# a = 2; b = 60  # var = 7200  [0.8]  [0.7962]
# a = 1; b = 120 # var = 14400 [1]    [1]
solution
sum(set_one(solution))
