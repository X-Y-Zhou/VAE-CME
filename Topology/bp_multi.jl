using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances,Random
using DelimitedFiles, Plots

include("../utils.jl")

seed = 1
rng = Random.seed!(seed)
ρ_list = rand(rng,Uniform(0.1,1),50)
batchsize = length(ρ_list)
τ = 40
N = 80
# train_sol = readdlm("Topology/matrix_bd.csv")
train_sol = hcat([birth_death(N, ρ_list[i], τ) for i = 1:length(ρ_list)]...)

# model initialization
model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> x .+ [i/τ for i in 1:N-1], x -> relu.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1)

#CME
function f1!(x,p,ρ)
    NN = re(p)(x)
    return vcat(-ρ*x[1] + NN[1]*x[2],
                [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
                sum(x)-1)
end

#solve P
P_0_distribution_list = [Poisson(ρ_list[i]*τ) for i=1:batchsize]
P_0_list = [[pdf(P_0_distribution_list[j],i) for i=0:N-1] for j=1:batchsize]
sol(p,ρ,P0) = nlsolve(x->f1!(x,p,ρ),P0).zero

@time sol_cme = hcat([sol(p1,ρ_list[i],P_0_list[i]) for i=1:batchsize]...);
mse = Flux.mse(sol_cme,train_sol)

@time sol_cme = hcat(map(i -> sol(p1, ρ_list[i], P_0_list[i]), 1:batchsize)...);
mse = Flux.mse(sol_cme, train_sol)

function loss_func(p)
    sol_cme = hcat([sol(p,ρ_list[i],P_0_list[i]) for i=1:batchsize]...)
    mse = Flux.mse(sol_cme,train_sol)
    loss = mse
    return loss
end

#check λ if is appropriate
@time mse = loss_func(p1)
mse_min = [mse]
@time grads = gradient(()->loss_func(p1) , ps)

epochs_all = 0

# training
lr_list = [0.01]  #lr需要操作一下的

lr_list = [0.01,0.008,0.006,0.004,0.002,0.001]

for lr in lr_list
    opt= ADAM(lr);
    epochs = 50
    print("learning rate = ",lr,"\n")

    @time for epoch in 1:epochs
        print(epoch,"\n")
        grads = gradient(()->loss_func(p1) , ps)
        Flux.update!(opt, ps, grads)

        mse = loss_func(p1)
        if mse<mse_min[1]
            df = DataFrame(p1 = p1)
            CSV.write("Topology/params_trained_bp.csv",df)
            mse_min[1] = mse
        end
        print(mse,"\n")
    end
end

using CSV,DataFrames
df = CSV.read("Topology/params_trained_bp.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);
# end

solution = hcat([sol(p1,ρ_list[i],P_0_list[i]) for i=1:batchsize]...)
mse = Flux.mse(solution,train_sol)

plot(0:N-1,solution,linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,train_sol,linewidth = 3,label="exact",title="steady-state",line=:dash)

function plot_distribution(set)
    plot(0:N-1,solution[:,set],linewidth = 3,label="NN-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol[:,set],linewidth = 3,label="exact",title=join(["ρ=",ρ_list[set]]),line=:dash)
end
plot_distribution(1)

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
plot_channel(1)

for i = 1:5
    p = plot_channel(i)
    savefig(p,"Topology/train_results/fig_$i.svg")
end