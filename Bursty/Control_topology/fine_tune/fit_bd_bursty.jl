using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

N = 65
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

function birth_death(ρ,t,N)
    distribution = Poisson(ρ*t)
    P = zeros(N)
    for i=1:N
        P[i] = pdf(distribution,i)
    end
    return P
end;

ρ = 0.0282*3.46
a = 0.0282
b = 3.46
τ = 120

train_sol_bd = birth_death(ρ,τ,N)
train_sol_bursty = bursty(N,a,b,τ)

# model 
model = Chain(Dense(N, 500, tanh), Dense(500, 4), x ->exp.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);

NN_list_bd = []
function f_bd!(x,p)
    l,m,n,o = re(p)(x)
    NN = f_NN.(1:N-1,l,m,n,o)
    push!(NN_list_bd,NN)
    return vcat(-ρ*x[1] + NN[1]*x[2],
                [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
                sum(x)-1)
end

NN_list_bursty = []
function f_bursty!(x,p)
    l,m,n,o = re(p)(x)
    NN = f_NN.(1:N-1,l,m,n,o)
    push!(NN_list_bursty,NN)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

P_0_bd = [pdf(Poisson(ρ*τ),j) for j=0:N-1]
sol_bd(p1,P_0) = nlsolve(x->f_bd!(x,p1),P_0).zero
solution_bd = sol_bd(p1,P_0)

P_0_bursty = [pdf(NegativeBinomial(a*τ, 1/(1+b)),j) for j=0:N-1]
sol_bursty(p1,P_0) = nlsolve(x->f_bursty!(x,p1),P_0).zero
solution_bursty = sol_bursty(p1,P_0_bursty)

plot(NN_list_bd[end],label="NN-bd")
plot!([0,60],[0,0.5],label="y=x/tau")

plot(NN_list_bursty[end],label="NN-bursty")
plot!(P[2:N],label="exact-NN")

function loss_func_bd(p)
    solution = sol_bd(p,P_0)
    mse = Flux.mse(solution,train_sol_bd)
    loss = mse
    return loss
end

function loss_func_bursty(p)
    solution = sol_bursty(p,P_0)
    mse = Flux.mse(solution,train_sol_bursty)
    loss = mse
    return loss
end

function loss_func(p)
    loss = loss_func_bd(p)+loss_func_bursty(p)
    return loss
end

loss_func(p1)
@time grads = gradient(()->loss_func(p1) , ps)

# training
lr = 0.006;  #lr需要操作一下的

lr_list = [0.01,0.008,0.006,0.004,0.002,0.001]

# for lr in lr_list
opt= ADAM(lr);
epochs = 20
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    print(epoch,"\n")
    grads = gradient(()->loss_func(p1) , ps)
    Flux.update!(opt, ps, grads)

    mse = loss_func(p1)
    if mse<mse_min[1]
        df = DataFrame(p1 = p1)
        CSV.write("Bursty/Control_topology/fine_tune/fit_bd_bursty.csv",df)
        mse_min[1] = mse
    end
    push!(mse_list,mse)
    print(mse,"\n")
end

using CSV,DataFrames
df = CSV.read("Bursty/Control_topology/fine_tune/fit_bd_bursty.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

mse_min = [7.724184741696977e-5]

solution_bd = sol_bd(p1,P_0)
mse_bd = Flux.mse(solution_bd,train_sol_bd)

solution_bursty = sol_bursty(p1,P_0)
mse_bursty = Flux.mse(solution_bursty,train_sol_bursty)
mse = mse_bd + mse_bursty

plot(0:N-1,solution_bd,linewidth = 3,label="NN-CME",xlabel = "# of products \n", ylabel = "\n Probability")
plot!(0:N-1,train_sol_bd,linewidth = 3,label="birth_death",line=:dash)

plot(0:N-1,solution_bursty,linewidth = 3,label="NN-CME",xlabel = "# of products \n", ylabel = "\n Probability")
plot!(0:N-1,train_sol_bursty,linewidth = 3,label="bursty",line=:dash)


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
