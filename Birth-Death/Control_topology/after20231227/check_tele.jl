using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

# topo tele
sigma_on = 0.003
sigma_off = 0.004
rho_on = 0.3
rho_off = 0.0
gamma= 0.0

a = sigma_on
b = rho_on/sigma_off
# N = 70

# check_sol = vec(readdlm("Birth-Death/Control_topology/after20231211-2/ssa_tele.csv", ',')[2:N+1,:])
# model
τ = 120
N = 100

# model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> 0.03.*x .+ [i/τ for i in 1:N-1], x -> relu.(x));
model = Chain(Dense(N, 10, tanh), Dense(10, 5), x -> exp.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);

using CSV,DataFrames
# NN
# df = CSV.read("Birth-Death/Control_topology/after20231222/params_trained_bp.csv",DataFrame)

# function
df = CSV.read("Birth-Death/Control_topology/after20231224/params_trained_bp-2_1.csv",DataFrame)

p1 = df.p1
ps = Flux.params(p1);

function f1!(x,p,sigma_on,sigma_off,rho_on,set)
    NN1 = re(p)(x[1:N])
    l,m,n,o,k = re(p)(x[1:N])
    NN1 = f_NN.(1:N,l,m,n,o,k/τ)

    push!(NN1_list_all[set],NN1)
    # NN1 = re(p)(x[1:N].*((sigma_on+sigma_off)/sigma_on))
    # l,m,n,o = re(p)(z)
    # NN1 = f_NN.(1:N-1,l,m,n,o)

    NN2 = re(p)(x[N+1:2*N])
    l,m,n,o,k = re(p)(x[N+1:2*N])
    NN2 = f_NN.(1:N,l,m,n,o,k/τ)

    push!(NN2_list_all[set],NN2)
    # NN2 = re(p)(x[N+1:2*N].*((sigma_on+sigma_off)/sigma_off))
    # l,m,n,o = re2(p2)(z)
    # NN2 = f_NN.(1:N-1,l,m,n,o)

    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)
end

# P0,P1
function solve_tele(sigma_on,sigma_off,rho_on,set)
    P_0_distribution = Poisson(rho_on*τ*sigma_on)
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
    P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]

    sol(p,P_0) = nlsolve(x->f1!(x,p,sigma_on,sigma_off,rho_on,set),P_0).zero
    solution = sol(p1,P_0_split)
    solution = solution[1:N]+solution[N+1:2*N]
    return solution
end

# i = 1
# solve_tele(p_list[i][1],p_list[i][2],p_list[i][3],i)
# p_list
# plot(NN1,label="NN1-exact")
# plot!(NN1_list_all[i][end])

# plot(NN2,label="NN2-exact")
# plot!(NN2_list_all[i][end])

p0p1_list
NN2_list_all_exact
NN1_list_all_exact

NN1_list_all = [[] for i=1:15]
NN2_list_all = [[] for i=1:15]
solution_list = []
solution_list

for i = 1:15
    print(i,"\n")
    solution = solve_tele(p_list[i][1],p_list[i][2],p_list[i][3],i)
    push!(solution_list,solution)
end
solution_list

function  plot_distribution(set)
    p=plot(0:N-1,solution_list[set],linewidth = 3,label="topo",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N-1,p0p1_list[set],linewidth = 3,label="exact",line=:dash,title=join(["+-ρ=",p_list[set]]))
end

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
    p11 = plot_distribution(11)
    p12 = plot_distribution(12)
    p13 = plot_distribution(13)
    p14 = plot_distribution(14)
    p15 = plot_distribution(15)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
         size=(1200,1200),layout=(4,4))
end
plot_all()

# NN1
function  plot_distribution(set)
    p=plot(0:59,NN1_list_all[set][end][1:60],linewidth = 3,label="topo",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:59,NN1_list_all_exact[set][1:60],linewidth = 3,label="NN1-exact",line=:dash,title=join(["+-ρ=",p_list[set]]))
end

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
    p11 = plot_distribution(11)
    p12 = plot_distribution(12)
    p13 = plot_distribution(13)
    p14 = plot_distribution(14)
    p15 = plot_distribution(15)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
         size=(1200,1200),layout=(4,4))
end
plot_all()

function  plot_distribution(set)
    p=plot(0:59,NN2_list_all[set][end][1:60],linewidth = 3,label="topo",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:59,NN2_list_all_exact[set][1:60],linewidth = 3,label="NN2-exact",line=:dash,title=join(["+-ρ=",p_list[set]]))
end

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
    p11 = plot_distribution(11)
    p12 = plot_distribution(12)
    p13 = plot_distribution(13)
    p14 = plot_distribution(14)
    p15 = plot_distribution(15)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
         size=(1200,1200),layout=(4,4))
    # plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,size=(1500,600),layout=(2,5))
    # plot(p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,900),layout=(3,5))
end
plot_all()






