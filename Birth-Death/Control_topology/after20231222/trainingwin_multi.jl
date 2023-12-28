using Distributed,Pkg
addprocs(10)
nprocs()
workers()

@everywhere using Flux, DiffEqSensitivity, DifferentialEquations
@everywhere using Distributions, Distances
@everywhere using DelimitedFiles, Plots

@everywhere include("../../../utils.jl")

a = 0.0282;
b = 3.46;
@everywhere τ = 120;
@everywhere N = 100

@everywhere ab_list = [[0.01,2],[0.01,4],[0.01,8],[0.01,15],
           [0.0075,5],[0.0075,7],[0.0075,10],[0.0075,15],
           [0.005,5],[0.005,10],[0.005,12],[0.005,18],
           [0.0025,5],[0.0025,10],[0.0025,15],[0.0025,20]]
@everywhere l_ablist = length(ab_list)
@everywhere train_sol = [bursty(N,ab_list[i][1],ab_list[i][2],τ) for i=1:l_ablist]

# model initialization
@everywhere model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> 0.03.*x .+ [i/τ for i in 1:N-1], x -> relu.(x));
@everywhere p1, re = Flux.destructure(model);
@everywhere ps = Flux.params(p1);

#CME
@everywhere function f1!(x,p,a,b)
    NN = re(p)(x)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

#solve P
@everywhere P_0_list = [[pdf(NegativeBinomial(ab_list[i][1]*τ, 1/(1+ab_list[i][2])),j) for j=0:N-1] for i=1:l_ablist]
@everywhere sol(p,a,b,P0) = nlsolve(x->f1!(x,p,a,b),P0).zero

# @everywhere weight = 5
@everywhere function compute_mse(p,set)
    sol_cme = sol(p,ab_list[set][1],ab_list[set][2],P_0_list[set])
    mse = Flux.mse(sol_cme,train_sol[set])
    reg_mse = Flux.mse(sol_cme[1],train_sol[set][1])
    mse_all = mse+weight*reg_mse
    return mse_all
end

@everywhere function loss_func(p,set)
    return sum(pmap(i->compute_mse(p,i),1:set))/set
end

set = 16
@time loss_func(p1,set)
@time grads = gradient(()->loss_func(p1,set) , ps)


lr = 0.01;  #lr需要操作一下的

lr_list = [0.025,0.01,0.008,0.006,0.004,0.002,0.001]
lr_list = [0.01,0.008,0.006,0.004,0.002,0.001]
lr_list = [0.05,0.0025,0.0015,0.001,0.004,0.002]
lr_list = [0.008,0.006,0.004]

for lr in lr_list
using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231222/params_trained_bp-1.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

# # training

opt= ADAM(lr);
epochs = 200
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    print(epoch,"\n")
    grads = gradient(()->loss_func(p1,l_ablist) , ps)
    Flux.update!(opt, ps, grads)

    mse = loss_func(p1,l_ablist)
    if mse<mse_min[1]
        df = DataFrame(p1 = p1)
        CSV.write("Birth-Death/Control_topology/after20231222/params_trained_bp-1.csv",df)
        mse_min[1] = mse
    end
    
    push!(mse_list,mse)
    print(mse,"\n")
end
end

mse_min = [0.00126077716603936]
mse_min 

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231222/params_trained_bp-1.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

@everywhere weight = 0.5
mse = loss_func(p1,set)
@time solution = pmap(i->sol(p1,ab_list[i][1],ab_list[i][2],P_0_list[i]),1:l_ablist);

function plot_distribution(set)
    plot(0:N-1,solution[set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol[set],linewidth = 3,label="exact",title=join(["a,b,τ=",ab_list[set]]),line=:dash)
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
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,size=(1200,1200),layout=(4,4))
end
plot_all()
