# multi params
using Plots,NLsolve
using LinearAlgebra, Distributions, DifferentialEquations
include("../../../SSA_car_utils.jl")

τ = 10
N = 120

ps_list = readdlm("Topology_results/fix_delay/tele_data/ps_televfinal.txt")
batchsize = size(ps_list,2)
matrix_tele = zeros(N,batchsize)
matrix_tele_p0 = zeros(N,batchsize)
matrix_tele_p1 = zeros(N,batchsize)

@time for i = 1:batchsize
    print(i,"\n")
    sigma_on = ps_list[:,i][1]
    sigma_off = ps_list[:,i][2]
    rho_on = ps_list[:,i][3]
    p_0,p_1,p0p1 = tele_delay(N,sigma_on,sigma_off,rho_on,τ)

    matrix_tele_p0[:,i] = p_0
    matrix_tele_p1[:,i] = p_1
    matrix_tele[:,i] = p0p1
end
writedlm("Topology_results/fix_delay/tele_data/matrix_televfinal.txt",matrix_tele)

# function plot_distribution(set)
#     plot(0:N-1,matrix_tele[:,set],linewidth = 3,label="tele")
#     # plot!(0:N-1,matrix_degrade[:,1,1,set],linewidth = 3,label="degrade",line=:dash,title=round.(ps_list[set],digits=4))
# end
# plot_distribution(20)

# function plot_channel(i)
#     p1 = plot_distribution(1+10*(i-1))
#     p2 = plot_distribution(2+10*(i-1))
#     p3 = plot_distribution(3+10*(i-1))
#     p4 = plot_distribution(4+10*(i-1))
#     p5 = plot_distribution(5+10*(i-1))
#     p6 = plot_distribution(6+10*(i-1))
#     p7 = plot_distribution(7+10*(i-1))
#     p8 = plot_distribution(8+10*(i-1))
#     p9 = plot_distribution(9+10*(i-1))
#     p10 = plot_distribution(10+10*(i-1))
#     plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,layouts=(2,5),size=(1500,600))
# end
# plot_channel(5)

# for i = 1:5
#     p = plot_channel(i)
#     savefig(p,"Topologyv2/tele/data/compare/fig_$i.svg")
# end

