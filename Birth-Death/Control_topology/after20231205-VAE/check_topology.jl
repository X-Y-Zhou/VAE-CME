using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

# check 
train_sol = vec(readdlm("Birth-Death/Control_topology/after20231205-VAE/train_data_ssa2.csv", ',')[2:end,:])
ρ_list = vec(readdlm("Birth-Death/Control_topology/after20231205-VAE/p.csv", ',')[2:end,:])

Ex = P2mean(train_sol)
Dx = P2var(train_sol)

a = 2Ex^2/(Dx-Ex)τ
b = (Dx-Ex)/2Ex

N = 70
τ = 120
train_sol = train_sol[1:N]
ρ_list = vcat(ρ_list,zeros(N-length(ρ_list)))
ρ_list

# model initialization
latent_size = 2;
encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
decoder = Chain(Dense(latent_size, 10),Dense(10 , N-1),x-> 0.03.*x.+[i/τ  for i in 1:N-1],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231205-VAE/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

#CME
function f1!(x,p1,p2,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN = re2(p2)(z)

    return vcat(-sum(ρ_list)*x[1]+NN[1]*x[2],[sum(ρ_list[i-j]*x[j] for j in 1:i-1) - 
            (sum(ρ_list)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

#solve P
P_0_distribution = Poisson(mean(ρ_list[1:10]*120))
# P_0_distribution = NegativeBinomial(a*τ, 1/(1+b))
P_0 = [pdf(P_0_distribution,i) for i=0:N-1]

ϵ = zeros(latent_size)
sol(p1,p2,ϵ,P_0) = nlsolve(x->f1!(x,p1,p2,ϵ),P_0).zero

solution = sol(params1,params2,ϵ,P_0)
mse = Flux.mse(solution,train_sol)

plot(0:N-1,solution,linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,train_sol,linewidth = 3,label="SSA",title="steady-state",line=:dash)

# topo bursty
# exact
function bursty(N,a,b,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

Ex = P2mean(train_sol)
Dx = P2var(train_sol)

a = 2Ex^2/(Dx-Ex)τ
b = (Dx-Ex)/2Ex

N = 70
check_sol = bursty(N,a,b,τ)

# model
latent_size = 2;
encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
decoder = Chain(Dense(latent_size, 10),Dense(10 , N-1),x-> 0.03.*x.+[i/τ  for i in 1:N-1],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231205-VAE/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

function f1!(x,p1,p2,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN = re2(p2)(z)

    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
    (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
P_0 = [pdf(P_0_distribution,i) for i=0:N-1]

ϵ = zeros(latent_size)
sol(p1,p2,ϵ,P_0) = nlsolve(x->f1!(x,p1,p2,ϵ),P_0).zero

solution = sol(params1,params2,ϵ,P_0)
mse = Flux.mse(solution,check_sol)

plot(0:N-1,solution,linewidth = 3,label="topo-bursty",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,check_sol,linewidth = 3,label="bursty-exact",title="steady-state",line=:dash)

# topo tele
sigma_on = a
sigma_off = 1
rho_on = b
rho_off = 0.0
gamma= 0.0

a = sigma_on
b = rho_on/sigma_off
N = 70

check_sol = vec(readdlm("Birth-Death/Control_topology/after20231205-VAE/ssa_tele.csv", ',')[2:N+1,:])

# model
latent_size = 2;
encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
decoder = Chain(Dense(latent_size, 10),Dense(10 , N-1),x-> 0.03.*x.+[i/τ  for i in 1:N-1],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231205-VAE/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

# P0,P1
function f1!(x,p1,p2,ϵ)
    h = re1(p1)(x[1:N])
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN1 = re2(p2)(z)

    h = re1(p1)(x[N+1:2*N])
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN2 = re2(p2)(z)

    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)
end

P_0_distribution = Poisson(rho_on*τ*sigma_on)
P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]

ϵ = zeros(latent_size)
sol(p1,p2,ϵ,P_0) = nlsolve(x->f1!(x,p1,p2,ϵ),P_0).zero
solution = sol(params1,params2,ϵ,P_0_split)
solution = solution[1:N]+solution[N+1:2*N]
# mse = Flux.mse(solution,check_sol)

plot(solution,linewidth = 3,label="topo-tele",xlabel = "# of products \n", ylabel = "\n Probability")
plot!(0:N-1,check_sol,linewidth = 3,label="SSA",line=:dash)



