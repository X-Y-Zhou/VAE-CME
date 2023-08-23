using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

#exact solution
function bursty(N,τ)
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

N = 64
train_sol = bursty(N,120)

# model initialization
latent_size = 10;
encoder = Chain(Dense(N, 20,tanh),Dense(20, latent_size * 2));
decoder = Chain(Dense(latent_size, 20),Dense(20 , N-1),x->0.03.* x.+[i/τ  for i in 1:N-1],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

#CME
function f1!(x,p1,p2,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN = re2(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

#solve P
P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
P_0 = [pdf(P_0_distribution,i) for i=0:N-1]

ϵ = zeros(latent_size)
sol(p1,p2,ϵ) = nlsolve(x->f1!(x,p1,p2,ϵ),P_0).zero
sol(params1,params2,ϵ)

function loss_func(p1,p2,ϵ)
    sol_cme = sol(p1,p2,ϵ)
        
    mse = Flux.mse(sol_cme,train_sol)
    print(mse," ")

    μ, logσ = split_encoder_result(re1(p1)(sol_cme), latent_size)
    kl = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1 .- (2 .* logσ))
    print(kl," ")

    loss = λ*mse + kl
    print(loss,"\n")
    return loss
end

λ = 5000000

#check λ if is appropriate
ϵ = zeros(latent_size)
loss_func(params1,params2,ϵ)
grads = gradient(()->loss_func(params1,params2,ϵ) , ps)

epochs_all = 0

#training
lr = 0.006;  #lr需要操作一下的
opt= ADAM(lr);
epochs = 20
epochs_all = epochs_all + epochs
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    ϵ = rand(Normal(),latent_size)
    print(epoch,"\n")
    grads = gradient(()->loss_func(params1,params2,ϵ) , ps)
    Flux.update!(opt, ps, grads)

    ϵ = zeros(latent_size)
    solution = sol(params1,params2,ϵ)
    mse = Flux.mse(solution,train_sol)
    
    if mse<mse_min[1]
        df = DataFrame( params1 = params1,params2 = vcat(params2,[0 for i=1:length(params1)-length(params2)]))
        CSV.write("Bursty/Control_rate_Inference/params_ss.csv",df)
        mse_min[1] = mse
    end

    push!(mse_list,mse)
    print(Flux.mse(sol(params1,params2,0),train_sol),"\n")#这个大概到1e-6差不多拟合了
end

mse_list
mse_min 

mse_min = [0.00012192452342102719]

using CSV,DataFrames
df = CSV.read("Bursty/Control_rate_Inference/params_ss.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

ϵ = zeros(latent_size)
solution = sol(params1,params2,ϵ)
Flux.mse(solution,train_sol)

plot(0:N-1,solution,linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,train_sol,linewidth = 3,label="exact",title="steady-state",line=:dash)


#test and check
P_trained = sol(params1,params2,0)
bar(0:N-1, P_trained, label = "trained", xlabel = "# of products", ylabel = "Probability")
plot!(0:N-1, P_exact, linewidth = 3, label = "Exact solution")

#ok，下面都是做测试
truncation = 45
if length(P_trained)>truncation
    P_trained = P_trained[1:truncation]
else
    P_trained = vcat(P_trained,[0 for i=1:truncation-length(P_trained)])
end

mse = Flux.mse(P_trained,P_exact)

#params save
using DataFrames
using CSV
params1
params2
df = DataFrame( params1 = params1,params2 = vcat(params2,[0 for i=1:length(params1)-length(params2)]))
vcat(params2,[0 for i=1:length(params1)-length(params2)])
CSV.write("machine-learning//results//params_trained_SSA100000.csv",df)

#test Extenicity τ = 60
decoder_changed = Chain(decoder[1],decoder[2],x->0.03.* x.+[i/60  for i in 1:N-1],decoder[4]);
P_train_exten = bursty(N,60);

_,re2_changed = Flux.destructure(decoder_changed);

function f2!(x,p1,p2,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN = re2_changed(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end
sol_exten(p1,p2,ϵ) = nlsolve(x->f2!(x,p1,p2,ϵ),P_train).zero

P_trained_exten = sol_exten(params1,params2,0)
bar(0:length(P_trained_exten)-1, P_trained_exten, label = "trained", fmt=:svg, xlabel = "# of products", ylabel = "Probability")
plot!(0:length(P_trained_exten)-1, P_train_exten, linewidth = 3, label = "Exact solution")
