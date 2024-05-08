using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

a = 0.0282;
b = 3.46;
τ = 120;
N = 100

ab_list = [[0.0025,3],[0.0025,5],[0.0025,10],[0.0025,15],[0.0025,20],
           [0.005,3],[0.005,5],[0.005,10],[0.005,12],[0.005,18],
           [0.0075,3],[0.0075,5],[0.0075,7],[0.0075,10],[0.0075,15],
           [0.009,2],[0.009,4],[0.009,6],[0.009,8],[0.009,10],
           [0.01,1],[0.01,2],[0.01,4],[0.01,8],[0.01,10],

           [0.02,0.75],[0.02,1],[0.02,2],[0.02,6],[0.02,8],
           [0.024,0.75],[0.024,1],[0.024,2],[0.024,3],[0.024,4],
           [0.028,0.75],[0.028,1],[0.028,2],[0.028,3],[0.028,4],
           [0.032,0.75],[0.032,1],[0.032,2],[0.032,3],[0.032,4],
           [0.036,0.75],[0.036,1],[0.036,2],[0.036,3],[0.036,4],

           [0.04,0.75],[0.04,1],[0.04,2],[0.04,4],[0.04,6],
           [0.06,0.75],[0.06,1],[0.06,2],[0.06,3],[0.06,5],
           [0.08,0.75],[0.08,1],[0.08,2],[0.08,3],[0.08,4],
           [0.1,0.75],[0.1,1],[0.1,2],[0.1,3],[0.1,3.5],
           [0.25,0.5],[0.25,0.8],[0.25,1.25],[0.25,1.5],[0.25,1.75],

           [0.5,0.25],[0.5,0.5],[0.5,0.6],[0.5,0.8],[0.5,1.0],
           [0.75,0.2],[0.75,0.4],[0.75,0.5],[0.75,0.6],[0.75,0.75],
           [1.0,0.2],[1.0,0.3],[1.0,0.4],[1.0,0.5],[1.0,0.55],
           [1.25,0.1],[1.25,0.2],[1.25,0.3],[1.25,0.4],[1.25,0.45],
           [1.5,0.08],[1.5,0.1],[1.5,0.2],[1.5,0.3],[1.5,0.35],
           ]
l_ablist = length(ab_list)

# ab_list = ab_list./2
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
model = Chain(Dense(N, 10, tanh), Dense(10, 4), x -> exp.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);

# lmno_list = []
#CME
function out!(x,p)
    l,m,n,o = re(p)(x)
    # push!(lmno_list,[l,m,n,o])
    NN = f_NN.(1:N,l,m,n,o)
    return NN
end

add_length = 25
weight = 10
function loss_func(p)
    sol_cme = [out!(NN_input[i],p) for i=1:l_ablist]
    mse1 = sum(Flux.mse(sol_cme[i],NN_output[i]) for i=1:add_length)
    mse2 = sum(Flux.mse(sol_cme[i],NN_output[i]) for i=add_length+1:l_ablist)
    loss = (weight*mse1+mse2)/l_ablist
    return loss
end

@time loss_func(p1)
@time grads = gradient(()->loss_func(p1) , ps)

lr = 0.01;  #lr需要操作一下的

lr_list = [0.025,0.01,0.008,0.006,0.004,0.002,0.001]
lr_list = [0.01,0.01,0.008,0.008,0.006,0.004,0.002,0.001]
lr_list = [0.002,0.001]
lr_list = [0.0006,0.0004]
lr_list = [0.0003,0.00015,0.0001]
lr_list = [0.008,0.006,0.004]
lr_list = [0.008,0.006]
lr_list = [0.025]

for lr in lr_list
using CSV,DataFrames
df = CSV.read("Control_topology/after20231224/params_trained_bp-1_1.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

# # training

opt= ADAM(lr);
epochs = 2000
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    print(epoch,"\n")
    grads = gradient(()->loss_func(p1) , ps)
    Flux.update!(opt, ps, grads)

    mse = loss_func(p1)
    if mse<mse_min[1]
        df = DataFrame(p1 = p1)
        CSV.write("Control_topology/after20231224/params_trained_bp-1_1.csv",df)
        mse_min[1] = mse
    end
    
    push!(mse_list,mse)
    print(mse,"\n")
end
end

mse_min = [0.00021776781574262674]
mse_min 

using CSV,DataFrames
df = CSV.read("Control_topology/after20231224/params_trained_bp-1_1.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

solution = [out!(NN_input[i],p1) for i=1:l_ablist]
mse = sum(Flux.mse(solution[i],NN_output[i]) for i=1:l_ablist)/l_ablist

[Flux.mse(solution[i],NN_output[i]) for i=1:l_ablist]

lmno_list

function plot_distribution(set)
    plot(0:N-1,solution[set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,NN_output[set],linewidth = 3,label="exact",title=join(["a,b,τ=",ab_list[set]]),line=:dash)
end
plot_distribution(35)

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
    p25 = plot_distribution(25)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,
            p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,1500),layout=(5,5))
end
plot_all()

function plot_all()
    p1 = plot_distribution(26)
    p2 = plot_distribution(27)
    p3 = plot_distribution(28)
    p4 = plot_distribution(29)
    p5 = plot_distribution(30)
    p6 = plot_distribution(31)
    p7 = plot_distribution(32)
    p8 = plot_distribution(33)
    p9 = plot_distribution(34)
    p10 = plot_distribution(35)
    p11 = plot_distribution(36)
    p12 = plot_distribution(37)
    p13 = plot_distribution(38)
    p14 = plot_distribution(39)
    p15 = plot_distribution(40)
    p16 = plot_distribution(41)
    p17 = plot_distribution(42)
    p18 = plot_distribution(43)
    p19 = plot_distribution(44)
    p20 = plot_distribution(45)
    p21 = plot_distribution(46)
    p22 = plot_distribution(47)
    p23 = plot_distribution(48)
    p24 = plot_distribution(49)
    p25 = plot_distribution(50)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,
            p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,1500),layout=(5,5))
end
plot_all()

function plot_all()
    p1 = plot_distribution(51)
    p2 = plot_distribution(52)
    p3 = plot_distribution(53)
    p4 = plot_distribution(54)
    p5 = plot_distribution(55)
    p6 = plot_distribution(56)
    p7 = plot_distribution(57)
    p8 = plot_distribution(58)
    p9 = plot_distribution(59)
    p10 = plot_distribution(60)
    p11 = plot_distribution(61)
    p12 = plot_distribution(62)
    p13 = plot_distribution(63)
    p14 = plot_distribution(64)
    p15 = plot_distribution(65)
    p16 = plot_distribution(66)
    p17 = plot_distribution(67)
    p18 = plot_distribution(68)
    p19 = plot_distribution(69)
    p20 = plot_distribution(70)
    p21 = plot_distribution(71)
    p22 = plot_distribution(72)
    p23 = plot_distribution(73)
    p24 = plot_distribution(74)
    p25 = plot_distribution(75)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,
            p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,1500),layout=(5,5))
end
plot_all()

function plot_all()
    p1 = plot_distribution(76)
    p2 = plot_distribution(77)
    p3 = plot_distribution(78)
    p4 = plot_distribution(79)
    p5 = plot_distribution(80)
    p6 = plot_distribution(81)
    p7 = plot_distribution(82)
    p8 = plot_distribution(83)
    p9 = plot_distribution(84)
    p10 = plot_distribution(85)
    p11 = plot_distribution(86)
    p12 = plot_distribution(87)
    p13 = plot_distribution(88)
    p14 = plot_distribution(89)
    p15 = plot_distribution(90)
    p16 = plot_distribution(91)
    p17 = plot_distribution(92)
    p18 = plot_distribution(93)
    p19 = plot_distribution(94)
    p20 = plot_distribution(95)
    p21 = plot_distribution(96)
    p22 = plot_distribution(97)
    p23 = plot_distribution(98)
    p24 = plot_distribution(99)
    p25 = plot_distribution(100)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,
            p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,1500),layout=(5,5))
end
plot_all()