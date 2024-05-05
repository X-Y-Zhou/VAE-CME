using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances,Random
using DelimitedFiles, Plots

include("../utils.jl")

# tele params
ps_matrix = readdlm("Topology/tele/data/ps_tele.csv")
sigma_on_list = ps_matrix[1,:]
sigma_off_list = ps_matrix[2,:]
rho_on_list = ps_matrix[3,:]
rho_off = 0.0
gamma= 0.0

# bd params
ps_matrix = vec(readdlm("Topology/ps_bd.csv"))
ρ_list = ps_matrix

τ = 40
N = 80
batchsize = length(ρ_list)

train_sol = hcat([birth_death(N, ρ_list[i], τ) for i = 1:length(ρ_list)]...)
check_sol = readdlm("Topology/tele/data/matrix_tele.csv")

# model initialization
model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> x .+ [i/τ for i in 1:N-1], x -> relu.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);

#CME
function f1!(x,p,ρ)
    NN = re(p)(x)
    return vcat(-ρ*x[1] + NN[1]*x[2],
                [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
                sum(x)-1)
end

sol_bd(p,ρ,P0) = nlsolve(x->f1!(x,p,ρ),P0).zero

function solve_bd(ρ,p)
    P_0_distribution = Poisson(ρ*τ)
    P_0 = [pdf(P_0_distribution,i) for i=0:N-1]
    solution = sol_bd(p,ρ,P_0)
    return solution
end

# @time solution_bd = hcat([solve_bd(ρ_list[i],p1) for i=1:batchsize]...)
@time solution_bd = hcat(pmap(i->solve_bd(ρ_list[i],p1),1:batchsize)...)
mse_bd = Flux.mse(solution_bd,train_sol)

function f_tele!(x,p,sigma_on,sigma_off,rho_on)
    NN1 = re(p)(x[1:N])
    NN2 = re(p)(x[N+1:2*N])

    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)
end

sol_tele(p,sigma_on,sigma_off,rho_on,P_0) = nlsolve(x->f_tele!(x,p,sigma_on,sigma_off,rho_on),P_0).zero

# P0,P1
function solve_tele(sigma_on,sigma_off,rho_on,p)
    P_0_distribution = Poisson(rho_on*τ*sigma_on)
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
    P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]

    solution = sol_tele(p,sigma_on,sigma_off,rho_on,P_0_split)
    solution = solution[1:N]+solution[N+1:2*N]
    return solution
end

using CSV,DataFrames
df = CSV.read("Topology/params_trained_bp.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

# @time solution_tele = hcat([solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],p1) for i=1:batchsize]...)
@time solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],p1),1:batchsize)...)
mse_tele = Flux.mse(solution_tele,check_sol)