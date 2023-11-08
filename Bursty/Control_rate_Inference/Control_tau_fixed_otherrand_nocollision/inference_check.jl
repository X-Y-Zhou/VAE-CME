using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots,CSV,DataFrames

include("/Users/x-y-zhou/Documents/GitHub/VAE-CME/utils.jl")

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
df = CSV.read("/Users/x-y-zhou/Documents/GitHub/VAE-CME/Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_nocollision/params_tfo.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length_2]

P_0_distribution_Extenicity = NegativeBinomial(a*τ, 1/(1+b));
P_0_Extenicity = [pdf(P_0_distribution_Extenicity,i) for i=0:N-1]
sol_Extenicity(p1,p2,a,b,Attribute,ϵ,P_0_Extenicity) = nlsolve(x->f_Extenicity!(x,p1,p2,a,b,Attribute,ϵ),P_0_Extenicity).zero

Attribute = 1
ϵ = zeros(latent_size)
solution = sol_Extenicity(params1,params2,a,b,Attribute,ϵ,P_0_Extenicity)

function check_inference(kinetic_params)
    a = kinetic_params[1]
    b = kinetic_params[2]
    Attribute = kinetic_params[3]

    ϵ = zeros(latent_size)
    P_0_distribution_Extenicity = NegativeBinomial(a*τ, 1/(1+b));
    P_0_Extenicity = [pdf(P_0_distribution_Extenicity,i) for i=0:N-1]
    solution = sol_Extenicity(params1,params2,a,b,Attribute,ϵ,P_0_Extenicity)
    return solution
end

width = "1.0-sqrt(6.0)" # 0.667
# width = "2.0-sqrt(4.0)" # 0.333
width
set = 1

kinetic_params_all = readdlm("/Users/x-y-zhou/Documents/GitHub/VAE-CME/Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_nocollision/infer_set$(set)_$(width)_exact_aa.csv",',')[2:end,:]
kinetic_params_all = kinetic_params_all[1:3,:]
SSA_data = readdlm("/Users/x-y-zhou/Documents/GitHub/VAE-CME/Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_nocollision/data/set$set/$width.csv",',')[2:end,:]

dataset = 1
kinetic_params = kinetic_params_all[:,dataset]
kinetic_params = [0.0282,3.46,0.667]
solution_inference = check_inference(kinetic_params)
Flux.mse(solution_inference,SSA_data)

solution_inference_list = []
for dataset = 1:5
    print(dataset,"\n")
    kinetic_params = kinetic_params_all[:,dataset]
    solution_inference = check_inference(kinetic_params)
    push!(solution_inference_list,solution_inference)
end

function plot_one(set)
    plot(0:N-1,solution_inference_list[set],lw=3,label="inference")
    plot!(0:N-1,vec(SSA_data),lw=3,label="SSA",line=:dash)
end

function plot_all()
    p1 = plot_one(1)
    p2 = plot_one(2)
    p3 = plot_one(3)
    p4 = plot_one(4)
    p5 = plot_one(5)
    plot(p1,p2,p3,p4,p5,size=(1500,300),layouts=(1,5))
end
plot_all()
savefig("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_nocollision/inference_pre_set$(set)-$width.pdf")


dataset = 1
kinetic_params = kinetic_params_all[:,dataset]
solution_inference_1 = check_inference(kinetic_params)

dataset = 3
kinetic_params = [0.0282,3.46,0.667]
solution_inference_2 = check_inference(kinetic_params)

Flux.mse(solution_inference_1,solution_inference_2)

plot(0:N-1,solution_inference_1,lw=3,label="a b attr=0.0183,3.46,0.438")
plot!(0:N-1,solution_inference_2,lw=3,label="a b attr=0.0282,3.46,0.667",line=:dash)