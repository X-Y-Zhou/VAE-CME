using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

N = 65
function birth_death(ρ,t,N)
    distribution = Poisson(ρ*t)
    P = zeros(N)
    for i=1:N
        P[i] = pdf(distribution,i)
    end
    return P
end;

ρ = 0.0282*3.46
τ = 120

exact_data = birth_death(ρ,τ,N)
# plot(exact_data)

# model 
model = Chain(Dense(N, 500, tanh), Dense(500, N-1), x-> x.+[i/τ  for i in 2:N],x ->relu.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);

# NN = [i/τ  for i in 1:N-1]
NN_list = []
function f1!(x,p)
    NN = re(p)(x)
    push!(NN_list,NN)
    # NN = NN1
    # NN = [i/τ  for i in 2:N]
    return vcat(-ρ*x[1] + NN[1]*x[2],
                [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
                sum(x)-1)
    # return vcat(sum(x)-1,
    #             [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
    #             ρ*x[N-1] + (-ρ-NN[N-1])*x[N])

    # return vcat(sum(x)-1,
    #             [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N],
    #             ρ*x[N] + (-ρ-NN[N])*x[N+1])
end

P_0_distribution = Poisson(ρ*τ)
P_0 = [pdf(Poisson(ρ*τ),j) for j=0:N-1]

sol(p1,P0) = nlsolve(x->f1!(x,p1),P0).zero
solution = sol(p1,P_0)

x = P_0
p = p1
NN = re(p)(x)
NN1 = NN

plot(NN,label="NN")
plot!([0,60],[0,0.5],label="y=x/tau")
savefig("Bursty/Control_topology/checkbd.pdf")

function loss_func(p)
    solution = sol(p,P_0)
    mse = Flux.mse(solution,exact_data)
    loss = mse
    return loss
end

loss_func(p1)
@time grads = gradient(()->loss_func(p1) , ps)

# training
lr = 0.0004;  #lr需要操作一下的

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
        CSV.write("Bursty/Control_topology/check_birth_death-origin2.csv",df)
        mse_min[1] = mse
    end
    push!(mse_list,mse)
    print(mse,"\n")
end
# end

mse_min = [3.9978808733135076e-5]

using CSV,DataFrames
df = CSV.read("Bursty/Control_topology/check_birth_death-origin2.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

solution = sol(p1,P_0)
Flux.mse(solution,exact_data)

plot(NN_list[end],label="NN")
plot!([0,60],[0,0.5],label="y=x/tau")

plot(0:N-1,solution,linewidth = 3,label="NN-CME",xlabel = "# of products \n", ylabel = "\n Probability")
plot!(0:N-1,exact_data,linewidth = 3,label="exact",line=:dash)
savefig("Bursty/Control_topology/birth-death-check.pdf")
