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
# SSA_data = readdlm("Control_tau_fixed_otherrand/Inference_data/set1/30-210_1.csv",',')[2:end,:]

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
df = CSV.read("Control_tau_fixed_otherrand_erlang/params_ct2.csv",DataFrame)
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
width = "10-12"

@time for dataset = [1,2,3,4,5]
print(dataset,"\n")    
SSA_data = readdlm("Control_tau_fixed_otherrand_erlang/Inference_data/set$set/$(width)_$dataset.csv",',')[2:end,:]
SSA_timepoints = round.(Int, vec(SSA_data).*sample_size)

function LogLikelihood(kinetic_params)
    a = kinetic_params[1]
    b = kinetic_params[2]
    Attribute = kinetic_params[3]

    df = CSV.read("Control_tau_fixed_otherrand_erlang/params_ct2.csv",DataFrame)
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

Ex = P2mean(SSA_data)
Dx = P2var(SSA_data)

a_0 = 2Ex^2/(Dx-Ex)τ
b_0 = (Dx-Ex)/2Ex

kinetic_params0 = [0.0282,3.46,0.3230]
SRange = [(0,0.06),(0,6),(0,1)]
res = bboptimize(LogLikelihood,kinetic_params0 ; Method = :adaptive_de_rand_1_bin_radiuslimited, 
SearchRange = SRange, NumDimensions = 3, MaxSteps = 200) #参数推断求解
thetax = best_candidate(res) #优化器求解参数
# best_fitness(res)

α = thetax[1]
β = thetax[2]
Attribute = thetax[3]

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
CSV.write("Control_tau_fixed_otherrand_erlang/temp_3.csv",df)

function check_inference(kinetic_params)
    a = kinetic_params[1]
    b = kinetic_params[2]
    Attribute = kinetic_params[3]

    df = CSV.read("Control_tau_fixed_otherrand_erlang/params_ct2.csv",DataFrame)
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

dataset = 5
result_list[dataset]
solution_inference = check_inference(result_list[dataset])
SSA_data = vec(readdlm("Control_tau_fixed_otherrand_erlang/data/set$set/$(width).csv",',')[2:end,:])
Flux.mse(solution_inference_1,SSA_data)

plot(0:N-1,solution_inference,lw=3,label="inference")
plot!(0:N-1,SSA_data,lw=3,label="SSA",line=:dash)

dataset = 4
result_list[dataset]
solution_inference_1 = check_inference(result_list[dataset])

dataset = 1
result_list[dataset]
solution_inference_2 = check_inference(result_list[dataset])

plot(0:N-1,solution_inference_1,lw=3,label="inference_1")
plot!(0:N-1,solution_inference_2,lw=3,label="inference_2")

# Erlang(a,b)
# a = 30;b = 4   # var = 480   [0]    [0]
# a = 20;b = 6   # var = 720   [0.1]  [0.1192]
# a = 10;b = 12  # var = 1440  [0.2]  [0.3230]
# a = 5; b = 24  # var = 2880  [0.4]  [0.5268]
# a = 2; b = 60  # var = 7200  [0.8]  [0.7962]
# a = 1; b = 120 # var = 14400 [1]    [1]

inference_kinetics = readdlm("Control_tau_fixed_otherrand_erlang/inference_kinetic2.csv",',')[2:end,:]
SSA_data = vec(readdlm("Control_tau_fixed_otherrand_erlang/data/set1/10-12.csv",',')[2:end,:])

N = 81
solution_inference_all = zeros(N,20)

for set = 1:20
    print(set,"\n")
    inference_kinetic = inference_kinetics[:,set]
    solution_inference = check_inference(inference_kinetic)
    solution_inference_all[:,set] = solution_inference
end
solution_inference_all

using Flux
dataset = 5
SSA_data = readdlm("Control_tau_fixed_otherrand_erlang/Inference_data/set$set/$(width)_$dataset.csv",',')[2:end,:]
SSA_timepoints = round.(Int, vec(SSA_data).*sample_size)

# Flux.mse(SSA_data,solution_inference_all[:,dataset])
# Flux.mse(SSA_data,solution_inference_all[:,dataset+5])

solution = vec(readdlm("Control_tau_fixed_otherrand_erlang/data/set1/10-12.csv",',')[2:end,:])
solution = set_one(solution)
log_value = log.(solution)
loglikelihood_value = -sum(SSA_timepoints.*log_value)/sample_size
fitness_inference_all

function plot_one(set)
    plot(0:N-1,solution_inference_all[:,set],lw=3,label="inference")
    plot!(0:N-1,SSA_data,lw=3,label="SSA",line=:dash)
end

function plot_all()
    p1 = plot_one(1)
    p2 = plot_one(2)
    p3 = plot_one(3)
    p4 = plot_one(4)
    p5 = plot_one(5)
    p6 = plot_one(6)
    p7 = plot_one(7)
    p8 = plot_one(8)
    p9 = plot_one(9)
    p10 = plot_one(10)
    p11 = plot_one(11)
    p12 = plot_one(12)
    p13 = plot_one(13)
    p14 = plot_one(14)
    p15 = plot_one(15)
    p16 = plot_one(16)
    p17 = plot_one(17)
    p18 = plot_one(18)
    p19 = plot_one(19)
    p20 = plot_one(20)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,
    p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,size=(1500,1200),layout=(4,5))
end
plot_all()



fitness_inference_all = zeros(5,2)
width

@time for dataset = [1,2,3,4,5]
print(dataset,"\n")    
SSA_data = readdlm("Control_tau_fixed_otherrand_erlang/Inference_data/set$set/$(width)_$dataset.csv",',')[2:end,:]
# SSA_data = readdlm("Control_tau_fixed_otherrand_erlang/data/set1/10-12.csv",',')[2:end,:]
SSA_timepoints = round.(Int, vec(SSA_data).*sample_size)

function LogLikelihood(kinetic_params)
    a = kinetic_params[1]
    b = kinetic_params[2]
    Attribute = kinetic_params[3]

    df = CSV.read("Control_tau_fixed_otherrand_erlang/params_ct2.csv",DataFrame)
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

inference_kinetic = inference_kinetics[:,dataset]
print(inference_kinetic,"\n")
fitness_inference_all[dataset,1] = LogLikelihood(inference_kinetic)
end

fitness_inference_all

fitness_inference_all1


SSA_data = readdlm("Control_tau_fixed_otherrand_erlang/Inference_data/set1/10-12_5.csv",',')[2:end,:]
SSA_timepoints = round.(Int, vec(SSA_data).*sample_size)

function LogLikelihood(kinetic_params)
    a = kinetic_params[1]
    b = kinetic_params[2]
    Attribute = kinetic_params[3]

    df = CSV.read("Control_tau_fixed_otherrand_erlang/params_ct2.csv",DataFrame)
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

using OptimalTransport
using Distances
SSA_data1 = readdlm("Control_tau_fixed_otherrand_erlang/Inference_data/set$set/$(width)_1.csv",',')[2:end,:]
SSA_data1 = vec(Float64.(SSA_data1))

μ = Categorical(vcat(SSA_data1,1-sum(SSA_data1)))
ν = Categorical(vcat(solution,1-sum(solution)))
γ = ot_plan(sqeuclidean, μ, ν);
ot_cost(sqeuclidean, μ, ν)

