using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

train_sol = vec(readdlm("Birth-Death/Control_topology/after20231205-2/train_data_ssa2.csv", ',')[2:end,:])
ρ_list = vec(readdlm("Birth-Death/Control_topology/after20231205-2/p.csv", ',')[2:end,:])

N = 70
τ = 120
train_sol = train_sol[1:N]
ρ_list = vcat(ρ_list,zeros(N-length(ρ_list)))

# model initialization
model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> 0.03.*x .+ [i/τ for i in 1:N-1], x -> relu.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1)

#CME
function f1!(x,p)
    NN = re(p)(x)
    return vcat(-sum(ρ_list)*x[1]+NN[1]*x[2],[sum(ρ_list[i-j]*x[j] for j in 1:i-1) - 
                (sum(ρ_list)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

#solve P
P_0_distribution = Poisson(mean(ρ_list[1:10]*120))
P_0 = [pdf(P_0_distribution,i) for i=0:N-1]

sol(p,P0) = nlsolve(x->f1!(x,p),P0).zero
# sol(params1,params2,a,b,ϵ,P_0_list[25])

function loss_func(p)
    sol_cme = sol(p,P_0)
    mse = Flux.mse(sol_cme,train_sol)
    loss = mse
    # print(loss,"\n")
    return loss
end

loss_func(p1)
@time grads = gradient(()->loss_func(p1) , ps)

epochs_all = 0

# training
lr = 0.01;  #lr需要操作一下的

lr_list = [0.01,0.008,0.006,0.004,0.002,0.001]

# for lr in lr_list
opt= ADAM(lr);
epochs = 10
epochs_all = epochs_all + epochs
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    print(epoch,"\n")
    grads = gradient(()->loss_func(p1) , ps)
    Flux.update!(opt, ps, grads)

    mse = loss_func(p1)
    if mse<mse_min[1]
        df = DataFrame(p1 = p1)
        CSV.write("Birth-Death/Control_topology/after20231205-2/params_trained_bp.csv",df)
        mse_min[1] = mse
    end
    
    push!(mse_list,mse)
    print(mse,"\n")
end

mse_list
mse_min 

mse_min = [2.353042372288063e-5]

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231205-2/params_trained_bp.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);
# end

solution = sol(p1,P_0)
mse = Flux.mse(solution,train_sol)

plot(0:N-1,solution,linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,train_sol,linewidth = 3,label="exact",title="steady-state",line=:dash)

