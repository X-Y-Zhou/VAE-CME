using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

train_solnet = readdlm("Birth-Death/Control_topology/after20231210/train_data_ssa2.csv", ',')[2:end,:]
ρ_listnet = readdlm("Birth-Death/Control_topology/after20231210/p.csv", ',')[2:end,:]

N = 70
τ = 120

train_sol = []
for i = 1:size(train_solnet,2)
    push!(train_sol,train_solnet[:,i][1:N])
end
train_sol[1]
plot(train_sol)

ρ_list_all = []
for i = 1:size(ρ_listnet,2)
    push!(ρ_list_all,vcat(ρ_listnet[:,i],zeros(N-length(ρ_listnet[:,i]))))
end
ρ_list_all[1]

lρ_list = length(ρ_list_all)


# model initialization
latent_size = 2;
encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
decoder = Chain(Dense(latent_size, 10),Dense(10 , N-1),x-> 0.03.*x.+[i/τ  for i in 1:N-1],x ->relu.(x));
# decoder = Chain(Dense(latent_size, 10),Dense(10 , 4),x-> exp.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

params1
params2

#CME
function f1!(x,p1,p2,ρ_list,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN = re2(p2)(z)

    # l,m,n,o = re2(p2)(z)
    # NN = f_NN.(1:N-1,l,m,n,o)

    return vcat(-sum(ρ_list)*x[1]+NN[1]*x[2],[sum(ρ_list[i-j]*x[j] for j in 1:i-1) - 
            (sum(ρ_list)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

#solve P
P_0_distribution_list = [Poisson(mean(ρ_list[i][1:10]*120)) for i=1:lρ_list]
P_0_list = [[pdf(P_0_distribution_list[i],j) for j=0:N-1] for i =1:lρ_list]
P_0_list[1]

i = 1
ϵ = zeros(latent_size)
sol(p1,p2,ρ_list,ϵ,P_0) = nlsolve(x->f1!(x,p1,p2,ρ_list,ϵ),P_0).zero
sol(params1,params2,ρ_list[i],ϵ,P_0_list[i])

function loss_func(p1,p2,ϵ)
    sol_cme = [sol(p1,p2,ρ_list[i],ϵ,P_0_list[i]) for i=1:lρ_list]
        
    mse = sum(Flux.mse(sol_cme[i],train_sol[i]) for i=1:lρ_list)/lρ_list
    print(mse," ")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[i]), latent_size) for i=1:lρ_list]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:lρ_list])/lρ_list
    print(kl," ")

    loss = λ*mse + kl
    print(loss,"\n")
    return loss
end

λ = 50000

#check λ if is appropriate
ϵ = zeros(latent_size)
@time loss_func(params1,params2,ϵ)
@time grads = gradient(()->loss_func(params1,params2,ϵ) , ps)

epochs_all = 0

#training
lr = 0.05;  #lr需要操作一下的
opt= ADAM(lr);
epochs = 10
epochs_all = epochs_all + epochs
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    ϵ = rand(Normal(),latent_size)
    print(epoch,"\n")
    grads = gradient(()->loss_func(params1,params2,ϵ) , ps)
    Flux.update!(opt, ps, grads)
    
    ϵ = zeros(latent_size)
    solution = [sol(params1,params2,ρ_list[i],ϵ,P_0_list[i]) for i=1:lρ_list]
    mse = sum(Flux.mse(solution[i],train_sol[i]) for i=1:lρ_list)/lρ_list

    if mse<mse_min[1]
        df = DataFrame(params1 = vcat(params1,[0 for i=1:length(params2)-length(params1)]),params2 = params2)
        CSV.write("Birth-Death/Control_topology/after20231210/params_trained.csv",df)
        mse_min[1] = mse
    end

    push!(mse_list,mse)
    print(mse,mse,"\n")
end

mse_list
mse_min 

mse_min = [0.00038168249677214273]

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231210/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

ϵ = zeros(latent_size)
solution = [sol(params1,params2,ρ_list[i],ϵ,P_0_list[i]) for i=1:lρ_list]
mse = sum(Flux.mse(solution[i],train_sol[i]) for i=1:lρ_list)/lρ_list

plot(0:N-1,solution,linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,train_sol,linewidth = 3,label="exact",title="steady-state",line=:dash)


