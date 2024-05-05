using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

# topo tele
sigma_on = 0.003
sigma_off = 0.004
rho_on = 0.3
rho_off = 0.0
gamma= 0.0

check_sol = readdlm("Topology/tele/data/matrix_tele.csv")
τ = 40
N = 80

model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> x .+ [i/τ for i in 1:N-1], x -> relu.(x));
# model = Chain(Dense(N, 10, tanh), Dense(10, 5), x -> exp.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);

using CSV,DataFrames
df = CSV.read("Topology/params_trained_bp.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

function f1!(x,p,sigma_on,sigma_off,rho_on)
    NN1 = re(p)(x[1:N])
    # NN1 = NN1 .* (sigma_off/(sigma_on+sigma_off))
    # NN1 = re(p)(x[1:N].*(sigma_off/(sigma_on+sigma_off)))
    # NN1 = NN1 .* ((sigma_on+sigma_off)/sigma_off)

    NN2 = re(p)(x[N+1:2*N])
    # NN2 = NN2 .* (sigma_on/(sigma_on+sigma_off))
    # NN2 = re(p)(x[N+1:2*N].*(sigma_on/(sigma_on+sigma_off)))
    # NN2 = NN2 .* ((sigma_on+sigma_off)/sigma_on)

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

    sol(p,P_0) = nlsolve(x->f1!(x,p,sigma_on,sigma_off,rho_on),P_0).zero
    solution = sol(p1,P_0_split)
    solution = solution[1:N]+solution[N+1:2*N]
    return solution
end

i = 1
sol_temp = solve_tele(ps_list[i][1],ps_list[i][2],ps_list[i][3])

p=plot(0:N-1,sol_temp,linewidth = 3,label="topo",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,check_sol[:,i],linewidth = 3,label="exact",line=:dash,title=join(["+-ρ=",round.(ps_list[i],digits=3)]))


@time solution = hcat([solve_tele(ps_list[i][1],ps_list[i][2],ps_list[i][3]) for i=1:batchsize]...)
Flux.mse(solution,check_sol)
Flux.mse(solution,check_sol)

function plot_distribution(set)
    p=plot(0:N-1,solution[:,set],linewidth = 3,label="topo",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N-1,check_sol[:,set],linewidth = 3,label="exact",line=:dash,title=join(["+-ρ=",round.(ps_list[set],digits=3)]))
end
plot_distribution(4)

function plot_channel(i)
    p1 = plot_distribution(1+10*(i-1))
    p2 = plot_distribution(2+10*(i-1))
    p3 = plot_distribution(3+10*(i-1))
    p4 = plot_distribution(4+10*(i-1))
    p5 = plot_distribution(5+10*(i-1))
    p6 = plot_distribution(6+10*(i-1))
    p7 = plot_distribution(7+10*(i-1))
    p8 = plot_distribution(8+10*(i-1))
    p9 = plot_distribution(9+10*(i-1))
    p10 = plot_distribution(10+10*(i-1))
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,layouts=(2,5),size=(1500,600))
end
plot_channel(1)

for i = 1:5
    p = plot_channel(i)
    savefig(p,"Topology/topo_results/fig_$i.svg")
end





