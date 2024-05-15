include("../utils.jl")
include("SSA_car_utils.jl")

N = 120
τ = 100

# Uniform(T1,T2)
T1T2_list = [[0,200],[25,175],[50,150],[75,125],[100,100],]

# bursty
for j = 1:5
    T1 = T1T2_list[j][1]
    T2 = T1T2_list[j][2]
    ps_matrix = readdlm("Topologyv3/ps_burstyv1.csv")
    batchsize = size(ps_matrix,2)
    matrix_bursty = zeros(N,batchsize)

    @time for i = 1:batchsize
        # print(i,"\n")
        t = 500
        n_cars_max = 30
        α = ps_matrix[:,i][1]
        β = ps_matrix[:,i][2]
        P_bursty = car_exact_bursty(T1,T2,α,β,t,n_cars_max,N)

        matrix_bursty[:,i] = P_bursty
    end

    writedlm("Topologyv3/bursty_data/matrix_bursty_$T1-$T2.csv",matrix_bursty)
end

ps_matrix = readdlm("Topologyv3/ps_burstyv1.csv")
batchsize_bursty = size(ps_matrix,2)
a_list = ps_matrix[1,:]
b_list = ps_matrix[2,:]
matrix_bursty = hcat([bursty_delay(N, a_list[i],b_list[i], τ) for i = 1:batchsize_bursty]...)
matrix_bursty_ori = readdlm("Topologyv3/bursty_data/matrix_bursty_100-100.csv")
Flux.mse(matrix_bursty,matrix_bursty_ori)

writedlm("Topologyv3/bursty_data/matrix_bursty_100-100v2.csv",matrix_bursty)

ps_matrix_bursty = readdlm("Topologyv3/ps_burstyv1.csv")
α,β = ps_matrix_bursty[:,50]
T1 = 100
T2 = 100
t = 500
n_cars_max = 30
plot(bursty_delay(N,α,β,τ),lw=3)
plot!(car_exact_bursty(T1,T2,α,β,100,n_cars_max,N),lw=3)
plot!(car_exact_bursty(T1,T2,α,β,500,n_cars_max,N),lw=3)
plot!(car_exact_bursty(T1,T2,α,β,1000,n_cars_max,N),lw=3)
plot!(car_exact_bursty(T1,T2,α,β,2000,n_cars_max,N),lw=3)
ps_matrix = readdlm("Topologyv3/ps_burstyv1.csv")

matrix_bursty_list = []
for j = 1:5
    T1 = T1T2_list[j][1]
    T2 = T1T2_list[j][2]
    matrix_bursty = readdlm("Topologyv3/bursty_data/matrix_bursty_$T1-$T2.csv")
    push!(matrix_bursty_list,matrix_bursty)
end
matrix_bursty_list

function plot_distribution(set)
    plot(0:N-1,matrix_bursty_list[1][:,set],linewidth = 3,line=:dash,title=round.(ps_matrix[:,set],digits=4))
    plot!(0:N-1,matrix_bursty_list[2][:,set],linewidth = 3,line=:dash)
    plot!(0:N-1,matrix_bursty_list[3][:,set],linewidth = 3,line=:dash)
    plot!(0:N-1,matrix_bursty_list[4][:,set],linewidth = 3,line=:dash)
    plot!(0:N-1,matrix_bursty_list[5][:,set],linewidth = 3,line=:dash)
end
plot_distribution(50)

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
plot_channel(5)

for i = 1:5
    p = plot_channel(i)
    savefig(p,"Topologyv3/bursty_data/compare/fig_$i.svg")
end

a_list = ps_matrix[1,:]
b_list = ps_matrix[2,:]
batchsize_bursty = size(ps_matrix,2)
train_sol = hcat([bursty_delay(N, a_list[i],b_list[i], τ) for i = 1:batchsize_bursty]...)

using Flux
Flux.mse(train_sol,matrix_bursty)
