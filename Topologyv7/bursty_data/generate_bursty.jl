include("../../utils.jl")
include("../../SSA_car_utils.jl")

N = 120
τ = exp(2)+100

μ_σ_list = [[0,sqrt(4)],[1,sqrt(2)]]

# bursty
for j = 1:2
    μ = μ_σ_list[j][1]
    σ = μ_σ_list[j][2]
    dist = LogNormal(μ,σ)+100
    ps_matrix = readdlm("Topologyv7/ps_burstyv1.csv")
    batchsize = size(ps_matrix,2)
    matrix_bursty = zeros(N,batchsize)

    @time for i = 1:batchsize
        # print(i,"\n")
        t = 500
        n_cars_max = 30
        α = ps_matrix[:,i][1]
        β = ps_matrix[:,i][2]
        P_bursty = car_exact_bursty(dist,α,β,t,n_cars_max,N)

        matrix_bursty[:,i] = P_bursty
    end

    writedlm("Topologyv7/bursty_data/matrix_bursty_μ=$μ.csv",matrix_bursty)
end

ps_matrix = readdlm("Topologyv7/ps_burstyv1.csv")
a_list = ps_matrix[1,:]
b_list = ps_matrix[2,:]
batchsize_bursty = size(ps_matrix,2)
matrix_bursty = hcat([bursty_delay(N, a_list[i],b_list[i], τ) for i = 1:batchsize_bursty]...)
writedlm("Topologyv7/bursty_data/matrix_bursty_μ=2.0.csv",matrix_bursty)


matrix_bursty_list = []
for j = 1:2
    μ = μ_σ_list[j][1]
    σ = μ_σ_list[j][2]
    matrix_bursty = readdlm("Topologyv7/bursty_data/matrix_bursty_μ=$μ.csv")
    push!(matrix_bursty_list,matrix_bursty)
end
matrix_bursty = readdlm("Topologyv7/bursty_data/matrix_bursty_μ=2.0.csv")
push!(matrix_bursty_list,matrix_bursty)

function plot_distribution(set)
    plot(0:N-1,matrix_bursty_list[1][:,set],linewidth = 3,line=:dash,title=round.(ps_matrix[:,set],digits=4))
    plot!(0:N-1,matrix_bursty_list[2][:,set],linewidth = 3,line=:dash)
    plot!(0:N-1,matrix_bursty_list[3][:,set],linewidth = 3,line=:dash)
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
    savefig(p,"Topologyv7/bursty_data/compare/fig_$i.svg")
end


ps_matrix = readdlm("Topologyv7/ps_burstyv1.csv")
batchsize = size(ps_matrix,2)
matrix_bursty = zeros(N,batchsize)

μ = 0
σ = sqrt(4)
dist = LogNormal(μ,σ)
mean(dist)

i = 10
α = ps_matrix[:,i][1]
β = ps_matrix[:,i][2]

N = 120
t = 500
n_cars_max = 50
P_bursty = car_exact_bursty(dist,α,β,t,n_cars_max,N)
P_mean = bursty_delay(N,α,β,mean(dist))

sum(P_bursty)
sum(P_mean)

plot(0:N-1,P_bursty,lw=3,label="0,sqrt(4)")
plot!(0:N-1,P_mean,lw=3,label="2,sqrt(0)",line=:dash)