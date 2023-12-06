using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

# check 
train_sol = vec(readdlm("Birth-Death/Control_topology/after20231205-2/train_data_ssa2.csv", ',')[2:end,:])
ρ_list = vec(readdlm("Birth-Death/Control_topology/after20231205-2/p.csv", ',')[2:end,:])

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
model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> 0.03.*x .+ [i/τ for i in 1:N-1], x -> relu.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1)

#CME
# NN_list = []
function f1!(x,p)
    NN = re(p)(x)
    # push!(NN_list,NN)
    return vcat(-sum(ρ_list)*x[1]+NN[1]*x[2],[sum(ρ_list[i-j]*x[j] for j in 1:i-1) - 
                (sum(ρ_list)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

#solve P
P_0_distribution = Poisson(mean(ρ_list[1:10]*120))
# P_0_distribution = NegativeBinomial(a*τ, 1/(1+b))

P_0 = [pdf(P_0_distribution,i) for i=0:N-1]
sol(p,P0) = nlsolve(x->f1!(x,p),P0).zero

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231205-2/params_trained_bp.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);
# end

solution = sol(p1,P_0)
mse = Flux.mse(solution,train_sol)

plot(0:N-1,solution,linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,train_sol,linewidth = 3,label="exact",title="steady-state",line=:dash)

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

check_sol = bursty(N,a,b,τ)

# model
model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> 0.03.*x .+ [i/τ for i in 1:N-1], x -> relu.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231205-2/params_trained_bp.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

function f1!(x,p)
    NN = re(p)(x)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
P_0 = [pdf(P_0_distribution,i) for i=0:N-1]
sol(p,P0) = nlsolve(x->f1!(x,p),P0).zero

solution = sol(p1,P_0)
mse = Flux.mse(solution,check_sol)

plot(0:N-1,solution,linewidth = 3,label="topo",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,check_sol,linewidth = 3,label="bursty-exact",title="steady-state",line=:dash)

# topo tele
sigma_on = a
sigma_off = 1.0
rho_on = b
rho_off = 0.0
gamma= 0.0

a = sigma_on
b = rho_on/sigma_off

check_sol = vec(readdlm("Birth-Death/Control_topology/after20231205-2/ssa_tele.csv", ',')[2:N+1,:])

# model
model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> 0.03.*x .+ [i/τ for i in 1:N-1], x -> relu.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231205-2/params_trained_bp.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

# # P0+P1
# function f1!(x,p)
#     x_excute = x[1:N]+x[N+1:2*N]
#     NN1 = re(p)(x_excute)
#     NN2 = re(p)(x_excute)

#     return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
#                 [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
#                 rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
#                 sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
#                 [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
#                 sum(x)-1)
# end

# P0,P1
function f1!(x,p)
    NN1 = re(p)(x[1:N])
    NN2 = re(p)(x[N+1:2*N])

    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)

    #     return vcat(sum(x)-1,
    #                 (-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
    #                 [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
    #                 rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
    #                 sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
    #                 [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)])
end

P_0_distribution = Poisson(rho_on*τ*sigma_on)
P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
# P_0 = exact_data
P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]
# plot(P_0_split)

sol(p,P0) = nlsolve(x->f1!(x,p),P0).zero

solution = sol(p1,P_0_split)
solution = solution[1:N]+solution[N+1:2*N]
# mse = Flux.mse(solution,check_sol)

plot(solution,linewidth = 3,label="topo",xlabel = "# of products \n", ylabel = "\n Probability")
plot!(0:N-1,check_sol,linewidth = 3,label="exact",line=:dash)


