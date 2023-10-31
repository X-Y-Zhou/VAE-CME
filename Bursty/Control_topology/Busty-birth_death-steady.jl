using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

N = 65
function birth_death(ρ,t,N)
    distribution = Poisson(ρ*t)
    P = zeros(N)
    for i=1:N
        P[i] = pdf(distribution,i)
    end
    return P
end;

ρ = 0.0282*3.46
τ = 120

exact_data = birth_death(ρ,τ,N)
plot(exact_data)

# model 
latent_size = 2;
encoder = Chain(Dense(N, 200,tanh),Dense(200, latent_size * 2));
decoder = Chain(Dense(latent_size, 200),Dense(200 , 4),x ->exp.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

using CSV,DataFrames
df = CSV.read("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_abcd/params_tfo2.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

p1 = params1
p2 = params2
x = P_0 = [pdf(Poisson(ρ*τ),j) for j=0:N-1]
h = re1(p1)(x)
μ, logσ = split_encoder_result(h, latent_size)
z = reparameterize.(μ, logσ, ϵ)
l,m,n,o = re2(p2)(z)

function f1!(x,p1,p2,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    l,m,n,o = re2(p2)(z)
    NN = f_NN.(1:N-1,l,m,n,o)
    # NN = re2(p2)(z)

    return vcat(-ρ*x[1] + NN[1]*x[2],
                [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
                sum(x)-1)
end

N/τ
P_0_distribution = Poisson(ρ*τ)
P_0 = [pdf(Poisson(ρ*τ),j) for j=0:N-1]

ϵ = zeros(latent_size)
sol(p1,p2,ϵ,P0) = nlsolve(x->f1!(x,p1,p2,ϵ),P0).zero

solution = sol(params1,params2,ϵ,P_0)

plot(0:N-1,solution,linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
plot!(0:N-1,exact_data,linewidth = 3,label="exact",line=:dash)
savefig("Bursty/Control_topology/birth-death.pdf")
