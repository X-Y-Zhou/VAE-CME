using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

# Load training data
train_sol = readdlm("Birth-Death/Control_topology/train_data.csv", ',')[2:end,:]
N = 80

train_sol_list = []
for i = 1:36
    push!(train_sol_list,vec(train_sol[:,i]))
end
plot(train_sol_list[1:18])
plot(train_sol_list[19:36])

# Exact solution
function birth_death(ρ,N,τ)
    distribution = Poisson(ρ*τ)
    P = zeros(N)
    for i=1:N
        P[i] = pdf(distribution,i-1)
    end
    return P
end;

τ = 120
ρ_0 = 0.282
ρ(X,α,k) = ρ_0*X^α/(k+X^α)+0.1

train_sol = train_sol[:,1:18]
α_list = [0.25,0.5,1]
k_list = [1,3,5,7,9,11]
p_list = [[α_list[i],k_list[j]] for i=1:length(α_list) for j=1:length(k_list)]
l_p_list = length(p_list)

# p_list[20]
# plot(train_sol[:,20])

# p_list = [[2.,3.]]
# l_p_list = length(p_list)
# train_sol = train_sol[:,20]

# Model initialization
latent_size = 2;
encoder = Chain(Dense(N, 200,tanh),Dense(200, latent_size * 2));
decoder = Chain(Dense(latent_size, 200),Dense(200 , 4),x ->exp.(x));
# decoder = Chain(Dense(latent_size, 200),Dense(200 , N-1),x-> x.+[i/τ  for i in 1:N-1],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

# Define the CME
function f1!(x,p1,p2,α,k,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)

    l,m,n,o = re2(p2)(z)
    NN = f_NN.(1:N-1,l,m,n,o)

    # NN = re2(p2)(z)

    return vcat(-ρ(0,α,k)*x[1] + NN[1]*x[2],
                [ρ(i-2,α,k)*x[i-1] + (-ρ(i-1,α,k)-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
                sum(x)-1)
end

P_0 = [pdf(Poisson(ρ_0*τ),j) for j=0:N-1]
# plot(P_0)
ϵ = zeros(latent_size)
sol(p1,p2,α,k,ϵ,P0) = nlsolve(x->f1!(x,p1,p2,α,k,ϵ),P0).zero

p_list
i = 1
p_list[i]

solution = sol(params1,params2,p_list[i][1],p_list[i][2],ϵ,P_0)

plot(solution)
plot!(train_sol[:,i])
# train_sol

Flux.mse(solution,train_sol[:,i])

function loss_func(p1,p2,ϵ)
    sol_cme = [sol(p1,p2,p_list[i][1],p_list[i][2],ϵ,P_0) for i=1:l_p_list]
    # sol_cme = set_one.(sol_cme)

    mse = sum(Flux.mse(sol_cme[i],train_sol[:,i]) for i=1:l_p_list)/l_p_list
    print(mse," ")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[i]), latent_size) for i=1:l_p_list]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:l_p_list])/l_p_list
    print(kl," ")

    loss = λ*mse + kl
    print(loss,"\n")
    return loss
end

λ = 10000000

#check λ if is appropriate
ϵ = zeros(latent_size)
loss_func(params1,params2,ϵ)
@time grads = gradient(()->loss_func(params1,params2,ϵ) , ps)

epochs_all = 0

lr_list = [0.01,0.008,0.006,0.004,0.002,0.001]
lr_list = [0.005,0.0025,0.0015,0.001]
lr_list = [0.006,0.004,0.002,0.001]
lr_list = [0.005,0.0025,0.0015,0.0008,0.0006]

lr_list = [0.0008,0.0006,0.0004]

lr = 0.001;  #lr需要操作一下的

# for lr in lr_list
using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/params_trained.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

# # training

opt= ADAM(lr);
epochs = 10
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    ϵ = rand(Normal(),latent_size)
    print(epoch,"\n")
    grads = gradient(()->loss_func(params1,params2,ϵ) , ps)
    Flux.update!(opt, ps, grads)

    ϵ = zeros(latent_size)
    solution = [sol(params1,params2,p_list[i][1],p_list[i][2],ϵ,P_0) for i=1:l_p_list]
    mse = sum(Flux.mse(solution[i],train_sol[:,i]) for i=1:l_p_list)/l_p_list

    if mse<mse_min[1]
        df = DataFrame( params1 = params1,params2 = vcat(params2,[0 for i=1:length(params1)-length(params2)]))
        CSV.write("Birth-Death/Control_topology/params_trained2.csv",df)
        mse_min[1] = mse
    end

    push!(mse_list,mse)
    print(mse,"\n")
end
# end

params1
params2

mse_min = [5.45207634569084e-7]
mse_min 

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/params_trained2.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

ϵ = zeros(latent_size)
solution = [sol(params1,params2,p_list[i][1],p_list[i][2],ϵ,P_0) for i=1:l_p_list]
mse = sum(Flux.mse(solution[i],train_sol[:,i]) for i=1:l_p_list)/l_p_list

function plot_distribution(set)
    plot(0:N-1,solution[set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol[:,set],linewidth = 3,label="exact",title=join(["α,k=",p_list[set]]),line=:dash)
end
plot_distribution(1)


# function plot_all()
#     p1 = plot_distribution(1)
#     p2 = plot_distribution(2)
#     p3 = plot_distribution(3)
#     p4 = plot_distribution(4)
#     plot(p1,p2,p3,p4,size=(600,600),layout=(2,2))
# end
# plot_all()
# # savefig("Control_rate_Inference/control_kinetic/fitting.svg")

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
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
            p16,p17,p18,size=(1800,900),layout=(3,6))
end
plot_all()