using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

# topo tele
# sigma_on = a
# sigma_off = 1
# rho_on = b
rho_off = 0.0
gamma= 0.0

# a = sigma_on
# b = rho_on/sigma_off
# N = 70

# check_sol = vec(readdlm("Birth-Death/Control_topology/after20231210/ssa_tele.csv", ',')[2:N+1,:])
# model
τ = 120
N = 70
latent_size = 2;
encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
decoder = Chain(Dense(latent_size, 10),Dense(10 , N-1),x-> 0.03.*x.+[i/τ  for i in 1:N-1],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231210/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

function f1!(x,p1,p2,sigma_on,sigma_off,rho_on,ϵ)
    h = re1(p1)(x[1:N])
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    # NN1 = re2(p2)(z)
    l,m,n,o = re2(p2)(z)
    NN1 = f_NN.(1:N-1,l,m,n,o)

    h = re1(p1)(x[N+1:2*N])
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    # NN2 = re2(p2)(z)
    l,m,n,o = re2(p2)(z)
    NN2 = f_NN.(1:N-1,l,m,n,o)

    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)
end

# P0,P1
function solve_tele(sigma_on,sigma_off,rho_on)
    P_0_distribution = Poisson(rho_on*τ*sigma_on)
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
    P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]

    ϵ = zeros(latent_size)
    sol(p1,p2,ϵ,P_0) = nlsolve(x->f1!(x,p1,p2,sigma_on,sigma_off,rho_on,ϵ),P_0).zero
    solution = sol(params1,params2,ϵ,P_0_split)
    solution = solution[1:N]+solution[N+1:2*N]
    return solution
end

a_list = [0.04,0.06,0.07,0.08,0.1]
b_list = [2,2.25,2.4,2.5,2.75]
ab_list = [[a_list[i],1.,b_list[j]] for i=1:length(a_list) for j=1:length(b_list)]
N = 70

a_list = [0.0082,0.0132,0.0182,0.0232,0.0282]
b_list = [1.46,1.96,2.46,2.96,3.46]
ab_list = [[a_list[i],1.,b_list[j]] for i=1:length(a_list) for j=1:length(b_list)]

solution_list = []
solution_list

for i = 12:25
    print(i,"\n")
    solution = solve_tele(ab_list[i][1],ab_list[i][2],ab_list[i][3])
    push!(solution_list,solution)
end

function  plot_distribution(set)
    p=plot(0:N-1,solution_list[set],linewidth = 3,label="topo",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N-1,train_sol_end_list[set],linewidth = 3,label="exact",line=:dash,title=join(["ab=",ab_list[set]]))
end
plot_distribution(1)

function plot_all()
    p1 = plot_distribution(1)
    p2 = plot_distribution(2)
    p3 = plot_distribution(3)
    p4 = plot_distribution(4)
    p5 = plot_distribution(5)
    p6 = plot_distribution(6)
    p7 = plot_distribution(7)
    p8 = plot_distribution(8)
    p9 = plot_distribution(9)
    p10 = plot_distribution(10)
    # # p11 = plot_distribution(11)
    p12 = plot_distribution(12-1)
    p13 = plot_distribution(13-1)
    p14 = plot_distribution(14-1)
    p15 = plot_distribution(15-1)
    p16 = plot_distribution(16-1)
    p17 = plot_distribution(17-1)
    p18 = plot_distribution(18-1)
    p19 = plot_distribution(19-1)
    p20 = plot_distribution(20-1)
    p21 = plot_distribution(21-1)
    p22 = plot_distribution(22-1)
    p23 = plot_distribution(23-1)
    p24 = plot_distribution(24-1)
    p25 = plot_distribution(25-1)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p12,p13,p14,p15,
         p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,1500),layout=(5,5))
    # plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,size=(1500,600),layout=(2,5))
    # plot(p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,900),layout=(3,5))
end
plot_all()
2.96*2
sigma_on,sigma_off,rho_on = [0.0182,1,1.63]
# sigma_on,sigma_off,rho_on = ab_list[11]
@time solution = solve_tele(sigma_on,sigma_off,rho_on)
set
N = 64
plot(0:N-1,solution,linewidth = 3,label="topo",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,train_sol_end_list[1][1:64],linewidth = 3,label="exact",line=:dash,title=join(["ab=",ab_list[set]]))




