using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

a = 0.0282;
b = 3.46;
τ = 120;
N = 100

# ab_list = [[0.0025,5],[0.0025,10],[0.0025,15],[0.0025,20],
#            [0.005,5],[0.005,10],[0.005,12],[0.005,18],
#            [0.0075,5],[0.0075,7],[0.0075,10],[0.0075,15],
#            [0.01,2],[0.01,4],[0.01,8],[0.01,10],
#            [0.02,1],[0.02,2],[0.02,6],[0.02,8],
#            [0.04,1],[0.04,2],[0.04,4],[0.04,6],
#            [0.06,1],[0.06,2],[0.06,3],[0.06,5],
#            [0.08,1],[0.08,2],[0.08,3],[0.08,4],
#            [0.1,1],[0.1,2],[0.1,3],[0.1,3.5],
#            [0.25,0.8],[0.25,1.25],[0.25,1.5],[0.25,1.75],
#            [0.5,0.5],[0.5,0.6],[0.5,0.8],[0.5,1.0],
#            ]

ab_list = [[0.0025,5]]

l_ablist = length(ab_list)

NN_input = [bursty(N,ab_list[i][1],ab_list[i][2],τ) for i=1:l_ablist]
NN_output = [NN_bursty(N,ab_list[i][1],ab_list[i][2],τ) for i=1:l_ablist]

range = 1:4
plot(NN_input[range],label=false)
plot(NN_output[range],label=false)

plot(NN_input,label=false)
plot(NN_output,label=false)

range = 5:8
plot(NN_input[range],label=false)
plot(NN_output[range],label=false,ylims=(0,0.3))

# model initialization
model = Chain(Dense(N, 10, tanh), Dense(10, 5), x -> exp.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);

#CME
# function out!(x,p)
#     l,m,n,o = re(p)(x)
#     # push!(lmno_list,[l,m,n,o])
#     NN = f_NN.(1:N,l,m,n,o)
#     return NN
# end

function out!(x,p)
    l,m,n,o,k = re(p)(x)
    # push!(lmno_list,[l,m,n,o])
    NN = f_NN.(1:N,l,m,n,o,k/τ)
    return NN
end

function loss_func(p)
    sol_cme = [out!(NN_input[i],p) for i=1:l_ablist]
    mse = sum(Flux.mse(sol_cme[i],NN_output[i]) for i=1:l_ablist)/l_ablist
    loss = mse
    return loss
end

@time loss_func(p1)
@time grads = gradient(()->loss_func(p1) , ps)

lr = 0.004;  #lr需要操作一下的

lr_list = [0.025,0.01,0.008,0.006,0.004]
lr_list = [0.01,0.008,0.006,0.004]
lr_list = [0.0006,0.0004]
lr_list = [0.0003,0.00015,0.0001]
lr_list = [0.008,0.006,0.004]

# for lr in lr_list
using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231224/params_trained_bp-3_1.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

# # training

opt= ADAM(lr);
epochs = 100
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    print(epoch,"\n")
    grads = gradient(()->loss_func(p1) , ps)
    Flux.update!(opt, ps, grads)

    mse = loss_func(p1)
    if mse<mse_min[1]
        df = DataFrame(p1 = p1)
        CSV.write("Birth-Death/Control_topology/after20231224/params_trained_bp-3_1.csv",df)
        mse_min[1] = mse
    end
    
    push!(mse_list,mse)
    print(mse,"\n")
end
# end

mse_min = [1.2728582680918452]
mse_min 

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231224/params_trained_bp-3_1.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

solution = [out!(NN_input[i],p1) for i=1:l_ablist]
mse = sum(Flux.mse(solution[i],NN_output[i]) for i=1:l_ablist)/l_ablist
[Flux.mse(solution[i],NN_output[i]) for i=1:l_ablist]

function plot_distribution(set)
    plot(0:N-1,solution[set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,NN_output[set],linewidth = 3,label="exact",title=join(["a,b,τ=",ab_list[set]]),line=:dash)
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
    p11 = plot_distribution(11)
    p12 = plot_distribution(12)
    p13 = plot_distribution(13)
    p14 = plot_distribution(14)
    p15 = plot_distribution(15)
    p16 = plot_distribution(16)
    p17 = plot_distribution(17)
    p18 = plot_distribution(18)
    p19 = plot_distribution(19)
    p20 = plot_distribution(20)
    p21 = plot_distribution(21)
    p22 = plot_distribution(22)
    p23 = plot_distribution(23)
    p24 = plot_distribution(24)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,
            p17,p18,p19,p20,p21,p22,p23,p24,size=(1200,1800),layout=(6,4))
end
plot_all()

function plot_all()
    p1 = plot_distribution(25)
    p2 = plot_distribution(26)
    p3 = plot_distribution(27)
    p4 = plot_distribution(28)
    p5 = plot_distribution(29)
    p6 = plot_distribution(30)
    p7 = plot_distribution(31)
    p8 = plot_distribution(32)
    p9 = plot_distribution(33)
    p10 = plot_distribution(34)
    p11 = plot_distribution(35)
    p12 = plot_distribution(36)
    p13 = plot_distribution(37)
    p14 = plot_distribution(38)
    p15 = plot_distribution(39)
    p16 = plot_distribution(40)
    p17 = plot_distribution(41)
    p18 = plot_distribution(42)
    p19 = plot_distribution(43)
    p20 = plot_distribution(44)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,
            p17,p18,p19,p20,size=(1500,1200),layout=(4,5))
end
plot_all()
