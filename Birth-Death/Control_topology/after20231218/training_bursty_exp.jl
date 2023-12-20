using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

#exact solution
function bursty(N,a,b,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

a = 0.0282;
b = 3.46;
τ = 120;
N = 100

ab_list = [[0.003,10],[0.003,20],[0.005,10],[0.005,15],[0.005,20],[0.008,10],[0.008,15],
            [0.01,15],[0.01,10]]
l_ablist = length(ab_list)
# a,b = ab_list[1]
# train_sol = bursty(N,a,b,τ)

train_sol = [bursty(N,ab_list[i][1],ab_list[i][2],τ) for i=1:l_ablist]
# plot(train_sol_end_list[end],lw=3)
# plot(train_sol[9:10],lw=3,line=:dash,label="bursty")
plot(bursty(N,0.02,3,τ))

# a,b = [0.1,0.1]
# plot(bursty(N,a,b,τ),lw=3)

# model initialization
model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> 0.03.*x .+ [i/τ for i in 1:N-1], x -> relu.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);

#CME
function f1!(x,p,a,b)
    NN = re(p)(x)
    # l,m,n,o = re(p)(z)
    # NN = f_NN.(1:N-1,l,m,n,o)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

#solve P
P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
P_0_list = [[pdf(NegativeBinomial(ab_list[i][1]*τ, 1/(1+ab_list[i][2])),j) for j=0:N-1] for i=1:l_ablist]
sol(p,a,b,P0) = nlsolve(x->f1!(x,p,a,b),P0).zero

function loss_func(p)
    sol_cme = [sol(p,ab_list[i][1],ab_list[i][2],P_0_list[i]) for i=1:l_ablist]
    mse = sum(Flux.mse(sol_cme[i],train_sol[i]) for i=1:l_ablist)/l_ablist
    loss = mse
    # print(loss,"\n")
    return loss
end

@time loss_func(p1)
@time grads = gradient(()->loss_func(p1) , ps)

lr = 0.01;  #lr需要操作一下的
lr_list = [0.01,0.008,0.006,0.004,0.002,0.001]

for lr in lr_list
using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231218/params_trained_exp.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

# # training

opt= ADAM(lr);
epochs = 50
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    print(epoch,"\n")
    grads = gradient(()->loss_func(p1) , ps)
    Flux.update!(opt, ps, grads)

    mse = loss_func(p1)
    if mse<mse_min[1]
        df = DataFrame(p1 = p1)
        CSV.write("Birth-Death/Control_topology/after20231218/params_trained_exp.csv",df)
        mse_min[1] = mse
    end
    
    push!(mse_list,mse)
    print(mse,"\n")
end
end

mse_min = [0.0013060164889193901]
mse_min 

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231218/params_trained_exp.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

solution = [sol(p1,ab_list[i][1],ab_list[i][2],P_0_list[i]) for i=1:l_ablist]
mse = sum(Flux.mse(solution[i],train_sol[i]) for i=1:l_ablist)/l_ablist

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
    # p10 = plot_distribution(10)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,size=(1500,600),layout=(2,5))
end
plot_all()
