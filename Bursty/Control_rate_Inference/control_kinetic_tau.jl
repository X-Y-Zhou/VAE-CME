using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

# mean = 120
# Uniform(20,220) var = 3333  [1,1]
# Uniform(40,200) var = 2133
# Uniform(60,180) var = 1200
# Uniform(80,160) var = 533
# Uniform(100,140) var = 133
# Uniform(120,120) var = 0    [1,0]

# mean = 30
# Uniform(5,55) var = 208     [0,1]
# Uniform(10,50) var = 133
# Uniform(15,45) var = 75
# Uniform(20,40) var = 33
# Uniform(25,35) var = 8
# Uniform(30,30) var = 0      [0,0]
 

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

# τ~Uniform(20,220) var = 3333  [1,1]
decoder_1 = Chain(Dense(latent_size, 20),Dense(20 , N-1),x->0.03.* x.+[i/120  for i in 1:N-1],x ->relu.(x));

# τ~Uniform(120,120) var = 0    [1,0]
decoder_2  = Chain(decoder_1[1],decoder_1[2],x->0.03.* x.+[i/120  for i in 1:N-1],decoder_1[4]);

# τ~Uniform(5,55) var = 208     [0,1]
decoder_3  = Chain(decoder_1[1],decoder_1[2],x->0.03.* x.+[i/30  for i in 1:N-1],decoder_1[4]);

# τ~Uniform(30,30) var = 0      [0,0]
decoder_4  = Chain(decoder_1[1],decoder_1[2],x->0.03.* x.+[i/30  for i in 1:N-1],decoder_1[4]);

params1, re1   = Flux.destructure(encoder);
params2, re2_1 = Flux.destructure(decoder_1);
      _, re2_2 = Flux.destructure(decoder_2);
      _, re2_3 = Flux.destructure(decoder_3);
      _, re2_4 = Flux.destructure(decoder_4);
ps = Flux.params(params1,params2);

#CME
function f1!(x,p1,p2,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,[1,1])
    NN = re2_1(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

function f2!(x,p1,p2,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,[1,0])
    NN = re2_2(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

function f3!(x,p1,p2,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,[0,1])
    NN = re2_3(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

function f4!(x,p1,p2,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,[0,0])
    NN = re2_4(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

# solve P
P_0_distribution_120 = NegativeBinomial(a*120, 1/(1+b));
P_0_120 = [pdf(P_0_distribution_120,i) for i=0:N-1]
sol_1(p1,p2,ϵ) = nlsolve(x->f1!(x,p1,p2,ϵ),P_0_120).zero
sol_2(p1,p2,ϵ) = nlsolve(x->f2!(x,p1,p2,ϵ),P_0_120).zero

P_0_distribution_30 = NegativeBinomial(a*30, 1/(1+b));
P_0_30 = [pdf(P_0_distribution_30,i) for i=0:N-1]
sol_3(p1,p2,ϵ) = nlsolve(x->f3!(x,p1,p2,ϵ),P_0_30).zero
sol_4(p1,p2,ϵ) = nlsolve(x->f4!(x,p1,p2,ϵ),P_0_30).zero

ϵ = zeros(latent_size)
sol_1(params1,params2,ϵ)
sol_2(params1,params2,ϵ)
sol_3(params1,params2,ϵ)
sol_4(params1,params2,ϵ)

function loss_func_1(p1,p2,ϵ)
    sol_cme = sol(p1,p2,ϵ)
        
    mse = Flux.mse(sol_cme,train_sol[:,1])
    print(mse," ")

    μ, logσ = split_encoder_result(re1(p1)(sol_cme), latent_size)
    kl = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1 .- (2 .* logσ))
    print(kl," ")

    loss = λ1*mse + kl
    print(loss,"\n")
    return loss
end

function loss_func_2(p1,p2,ϵ)
    sol_cme = sol(p1,p2,ϵ)
        
    mse = Flux.mse(sol_cme,train_sol[:,2])
    print(mse," ")

    μ, logσ = split_encoder_result(re1(p1)(sol_cme), latent_size)
    kl = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1 .- (2 .* logσ))
    print(kl," ")

    loss = λ2*mse + kl
    print(loss,"\n")
    return loss
end

function loss_func_3(p1,p2,ϵ)
    sol_cme = sol(p1,p2,ϵ)
        
    mse = Flux.mse(sol_cme,train_sol[:,3])
    print(mse," ")

    μ, logσ = split_encoder_result(re1(p1)(sol_cme), latent_size)
    kl = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1 .- (2 .* logσ))
    print(kl," ")

    loss = λ3*mse + kl
    print(loss,"\n")
    return loss
end

function loss_func_4(p1,p2,ϵ)
    sol_cme = sol(p1,p2,ϵ)
        
    mse = Flux.mse(sol_cme,train_sol[:,4])
    print(mse," ")

    μ, logσ = split_encoder_result(re1(p1)(sol_cme), latent_size)
    kl = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1 .- (2 .* logσ))
    print(kl," ")

    loss = λ4*mse + kl
    print(loss,"\n")
    return loss
end

function loss_func(p1,p2,ϵ)
    loss = loss_func_1(p1,p2,ϵ) + loss_func_2(p1,p2,ϵ)+
           loss_func_3(p1,p2,ϵ) + loss_func_4(p1,p2,ϵ)
    return loss
end

λ1 = 5000000
λ2 = 5000000
λ3 = 5000000
λ4 = 5000000

#check λ if is appropriate
ϵ = zeros(latent_size)
loss_func_1(params1,params2,ϵ)
loss_func_2(params1,params2,ϵ)
loss_func_3(params1,params2,ϵ)
loss_func_4(params1,params2,ϵ)
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
    solution_1 = sol_1(params1,params2,ϵ)
    solution_2 = sol_2(params1,params2,ϵ)
    solution_3 = sol_3(params1,params2,ϵ)
    solution_4 = sol_4(params1,params2,ϵ)

    mse_1 = Flux.mse(solution_1,train_sol[:,1])
    mse_2 = Flux.mse(solution_2,train_sol[:,2])
    mse_3 = Flux.mse(solution_3,train_sol[:,3])
    mse_4 = Flux.mse(solution_4,train_sol[:,4])
    mse = mse_1+mse_2+mse_3+mse_4
    
    if mse<mse_min[1]
        df = DataFrame( params1 = params1,params2 = vcat(params2,[0 for i=1:length(params1)-length(params2)]))
        CSV.write("Bursty/Control_rate_Inference/params_ss.csv",df)
        mse_min[1] = mse
    end

    push!(mse_list,mse)
    print(mse,"\n")#这个大概到1e-6差不多拟合了
end

mse_list
mse_min 

mse_min = [0.00012192452342102719]

using CSV,DataFrames
df = CSV.read("Bursty/Control_rate_Inference/params_ckt.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

ϵ = zeros(latent_size)
solution_1 = sol_1(params1,params2,ϵ)
solution_2 = sol_2(params1,params2,ϵ)
solution_3 = sol_3(params1,params2,ϵ)
solution_4 = sol_4(params1,params2,ϵ)

mse_1 = Flux.mse(solution_1,train_sol[:,1])
mse_2 = Flux.mse(solution_2,train_sol[:,2])
mse_3 = Flux.mse(solution_3,train_sol[:,3])
mse_4 = Flux.mse(solution_4,train_sol[:,4])
mse = mse_1+mse_2+mse_3+mse_4

solution = [solution_1,solution_2,solution_3,solution_4]

function plot_distribution(set)
    plot(0:N-1,solution[set],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N-1,train_sol[:,set],linewidth = 3,label="exact",title=join(["a,b,τ=",ab_list[set]]),line=:dash)
end

function plot_all()
    p1 = plot_distribution(1)
    p2 = plot_distribution(2)
    p3 = plot_distribution(3)
    p4 = plot_distribution(4)
    plot(p1,p2,p3,p4,size=(400,400))
end
plot_all()

function sol_Extenicity(τ,Attribute)
    decoder_Extenicity  = Chain(decoder_1[1],decoder_1[2],x->0.03.* x.+[i/τ for i in 1:N-1],decoder_1[4]);
    _,re2_Extenicity = Flux.destructure(decoder_Extenicity);

    function f_Extenicity!(x,p1,p2,ϵ)
        h = re1(p1)(x)
        μ, logσ = split_encoder_result(h, latent_size)
        z = reparameterize.(μ, logσ, ϵ)
        z = vcat(z,Attribute)
        NN = re2_Extenicity(p2)(z)
        return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
                (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
    end

    P_0_distribution_Extenicity = NegativeBinomial(a*τ, 1/(1+b));
    P_0_Extenicity = [pdf(P_0_distribution_Extenicity,i) for i=0:N-1]

    sol_Extenicity(p1,p2,ϵ) = nlsolve(x->f_Extenicity!(x,p1,p2,ϵ),P_0_Extenicity).zero

    P_trained_Extenicity = sol_Extenicity(params1,params2,zeros(latent_size))
    return P_trained_Extenicity
end
