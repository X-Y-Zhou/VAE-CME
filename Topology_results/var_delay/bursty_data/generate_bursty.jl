include("../../../utils.jl")
include("../../../SSA_car_utils.jl")

N = 120
τ = 10

# Uniform(T1,T2)
T1T2_list = [[0,20],[5,15]]

# bursty
for j = 1:2
    T1 = T1T2_list[j][1]
    T2 = T1T2_list[j][2]
    ps_matrix = readdlm("Topology_results/var_delay/bursty_data/ps_burstyv1.txt")
    batchsize = size(ps_matrix,2)
    matrix_bursty = zeros(N,batchsize)

    @time for i = 1:batchsize
        # print(i,"\n")
        t = 500
        n_cars_max = 30
        α = ps_matrix[:,i][1]
        β = ps_matrix[:,i][2]
        P_bursty = car_exact_bursty(Uniform(T1,T2),α,β,t,n_cars_max,N)

        matrix_bursty[:,i] = P_bursty
    end
    writedlm("Topology_results/var_delay/bursty_data/matrix_bursty_$T1-$T2.txt",matrix_bursty)
end

ps_matrix = readdlm("Topology_results/var_delay/bursty_data/ps_burstyv1.txt")
a_list = ps_matrix[1,:]
b_list = ps_matrix[2,:]
batchsize_bursty = size(ps_matrix,2)
matrix_bursty = hcat([bursty_delay(N, a_list[i],b_list[i], τ) for i = 1:batchsize_bursty]...)
writedlm("Topology_results/var_delay/bursty_data/matrix_bursty_10-10.txt",matrix_bursty)


ps_matrix = readdlm("Topology_results/var_delay/bursty_data/ps_burstyv1.txt")
matrix_bursty_list = []
for j = 1:2
    T1 = T1T2_list[j][1]
    T2 = T1T2_list[j][2]
    matrix_bursty = readdlm("Topology_results/var_delay/bursty_data/matrix_bursty_$T1-$T2.txt")
    push!(matrix_bursty_list,matrix_bursty)
end
matrix_bursty = readdlm("Topology_results/var_delay/bursty_data/matrix_bursty_10-10.txt")
push!(matrix_bursty_list,matrix_bursty)

# matrix_bursty_ori = readdlm("Topologyv6/bursty_data/matrix_bursty_5-15.txt")
# Flux.mse(matrix_bursty_list[2],matrix_bursty_ori)

function plot_distribution(set)
    plot(0:N-1,matrix_bursty_list[3][:,set],linewidth = 3,label="Uniform(10,10)")
    plot!(0:N-1,matrix_bursty_list[1][:,set],linewidth = 3,line=:dash,title=round.(ps_matrix[:,set],digits=4),label="Uniform(0,20)")
    # plot!(0:N-1,matrix_bursty_list[2][:,set],linewidth = 3,line=:dash,label="Uniform(5,15)")
    # plot!(0:N-1,matrix_bursty_list[4][:,set],linewidth = 3,line=:dash)
    # plot!(0:N-1,matrix_bursty_list[5][:,set],linewidth = 3,line=:dash)
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
plot_channel(4)

for i = 1:5
    p = plot_channel(i)
    savefig(p,"Topology_results/var_delay/bursty_data/compare/fig_$i.svg")
end



N = 120
τ = 10

# Uniform(T1,T2)
T1T2_list = [[0,20],[5,15]]

# bursty
dist = Uniform(0,20)
ps_matrix = readdlm("Topologyv6/ps_burstyv1.txt")
batchsize = size(ps_matrix,2)

i = 25
t = 500
n_cars_max = 30
α = ps_matrix[:,i][1]
β = ps_matrix[:,i][2]
P_bursty = car_exact_bursty(dist,α,β,t,n_cars_max,N)

plot(0:N-1,P_bursty,lw=3)

n_cars_max = 30
mean_value = [P2mean(car_exact_bursty(dist,α,β,t,n_cars_max,N)) for t=1:500]
plot(mean_value,lw=3,line=:dash)

ps_matrix = readdlm("Topologyv6/ps_burstyv1.txt")
batchsize = size(ps_matrix,2)
matrix_bursty = zeros(N,batchsize)

dist = Uniform(0,20)
@time for i = 1:batchsize
    # print(i,"\n")
    t = 500
    n_cars_max = 50
    α = ps_matrix[:,i][1]
    β = ps_matrix[:,i][2]
    P_bursty = car_exact_bursty(dist,α,β,t,n_cars_max,N)

    matrix_bursty[:,i] = P_bursty
end

using Flux
matrix_bursty_ori = readdlm("Topologyv6/bursty_data/matrix_bursty_0-20.txt")
Flux.mse(matrix_bursty,matrix_bursty_ori)
sum(matrix_bursty)

function plot_distribution(set)
    plot(0:N-1,matrix_bursty_ori[:,set],linewidth = 3,label="ori",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,matrix_bursty[:,set],linewidth = 3,label="now",line=:dash)
end
plot_distribution(30)

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
plot_channel(3)


