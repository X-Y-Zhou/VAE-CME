# 用bursty的数据外拓tele双峰+binomial的数据
# 其中τ均值不变，服从某个概率分布，用Attribute对方差进行调控，是否能外拓
# τ 服从均值为100的均分分布

using Plots,Random,Distributions,DelimitedFiles
seed = 1
rng = Random.seed!(seed)
a_list = [rand(rng,Uniform(0.01,0.05),25);rand(rng,Uniform(0.05,0.1),25);]
# b_list = [rand(rng,Uniform(1,3),25);rand(rng,Uniform(3,6),25)]
b_list = [rand(rng,Uniform(1,3),25);rand(rng,Uniform(3,5),25)]
ps_matrix = hcat([[a_list[i],b_list[i]] for i=1:length(a_list)]...)
τ = 100
N = 120
writedlm("Topologyv3/ps_burstyv1.csv",ps_matrix)


batchsize_bursty = size(ps_matrix,2)
train_sol = hcat([bursty_delay(N, a_list[i],b_list[i], τ) for i = 1:batchsize_bursty]...)

function plot_distribution(set)
    plot(0:N-1,train_sol[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix[:,set],digits=3)]))
end

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
plot_channel(2)

for i = 1:5
    p = plot_channel(i)
    savefig(p,"Topologyv3/tele/data/compare/fig_$i.svg")
end


