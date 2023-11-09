using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

function bursty(N,a,b,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

a = 0.0282
b = 3.46
τ = 120
N = 65

exact_data = bursty(N,a,b,τ)
# plot(exact_data)

# model 
model = Chain(Dense(N, 500, tanh), Dense(500, N-1), x-> 0.03.* x.+[i/τ  for i in 1:N-1],x ->relu.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);

# NN_list = []
function f1!(x,p)
    NN = re(p)(x)
    # push!(NN_list,NN)
    NN = P[2:N+1]
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
P_0 = [pdf(P_0_distribution,j) for j=0:N-1]

sol(p1,P0) = nlsolve(x->f1!(x,p1),P0).zero
solution = sol(p1,P_0)

function loss_func(p)
    solution = sol(p,P_0)
    mse = Flux.mse(solution,exact_data)
    loss = mse
    return loss
end

loss_func(p1)
@time grads = gradient(()->loss_func(p1) , ps)

# training
lr = 0.006;  #lr需要操作一下的

lr_list = [0.01,0.008,0.006,0.004,0.002,0.001]

# for lr in lr_list
opt= ADAM(lr);
epochs = 30
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    print(epoch,"\n")
    grads = gradient(()->loss_func(p1) , ps)
    Flux.update!(opt, ps, grads)

    mse = loss_func(p1)
    if mse<mse_min[1]
        df = DataFrame(p1 = p1)
        CSV.write("Control_topology/check_bursty-origin.csv",df)
        mse_min[1] = mse
    end
    push!(mse_list,mse)
    print(mse,"\n")
end
# end

mse_min = [0.00012069631994114625]

using CSV,DataFrames
df = CSV.read("Control_topology/check_bursty-origin.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

solution = sol(p1,P_0)
Flux.mse(solution,exact_data)

plot(NN_list[end],label="NN")
plot!(P[2:N+1],label="exact_NN")
plot!([0,60],[0,0.5],label="y=x/tau")

plot(0:N-1,solution,linewidth = 3,label="NN-CME",xlabel = "# of products \n", ylabel = "\n Probability")
plot!(0:N-1,exact_data,linewidth = 3,label="exact",line=:dash)
# savefig("Control_topology/birth-death-check.pdf")





f(u) = a*b*u/(1-b*(u-1))
g(u) = exp(a*b*τ*(u-1)/(1-b*(u-1)))
fg(u) = f(u)*g(u)

taylorexpand_fg = taylor_expand(x->fg(x),0,order=N+1)
taylorexpand_g = taylor_expand(x->g(x),0,order=N+1)
P = zeros(N+1)
for j in 1:N+1
    P[j] = taylorexpand_fg[j-1]/taylorexpand_g[j-1]
end
P
plot(P)