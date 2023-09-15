using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

# training data
# set1
# mean = 120
# α = 0.0282 β = 3.46 
# Uniform(0,240)   var = 4800  [1]
# Uniform(30,210)  var = 2700
# Uniform(60,180)  var = 1200
# Uniform(90,150)  var = 300
# Uniform(120,120) var = 0     [0]

# set2
# mean = 120
# α = 0.0082 β = 1.46 
# Uniform(0,240)   var = 4800  [1]
# Uniform(30,210)  var = 2700
# Uniform(60,180)  var = 1200
# Uniform(90,150)  var = 300
# Uniform(120,120) var = 0     [0]

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

N = 65

data = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/data/training_data.csv",',')[2:end,:]
train_sol_1 = data[:,[1,6]]
train_sol_2 = data[:,[5,10]]

ab_list = [[0.0282,3.46],[0.0082,1.46]]
l_ablist = length(ab_list)

# model initialization
latent_size = 10;
encoder = Chain(Dense(N, 20,tanh),Dense(20, latent_size * 2));
decoder_1 = Chain(Dense(latent_size+1, 20),Dense(20 , N-1),x->0.03.* x.+[i/τ  for i in 1:N-1],x ->relu.(x));
decoder_2  = Chain(decoder_1[1],decoder_1[2],x->0.03.* x.+[i/τ  for i in 1:N-1],decoder_1[4]);

params1, re1 = Flux.destructure(encoder);
params2, re2_1 = Flux.destructure(decoder_1);
      _, re2_2 = Flux.destructure(decoder_2);
ps = Flux.params(params1,params2);

#CME 4800~1
function f1!(x,p1,p2,a,b,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,1)
    NN = re2_1(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

# 0~0
function f2!(x,p1,p2,a,b,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,0)
    NN = re2_2(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

#solve P
P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
P_0_list = [[pdf(NegativeBinomial(ab_list[i][1]*τ, 1/(1+ab_list[i][2])),j) for j=0:N-1] for i=1:l_ablist]

ϵ = zeros(latent_size)
sol_1(p1,p2,a,b,ϵ,P0) = nlsolve(x->f1!(x,p1,p2,a,b,ϵ),P0).zero
sol_2(p1,p2,a,b,ϵ,P0) = nlsolve(x->f2!(x,p1,p2,a,b,ϵ),P0).zero

function loss_func_1(p1,p2,ϵ)
    sol_cme = [sol_1(p1,p2,ab_list[i][1],ab_list[i][2],ϵ,P_0_list[i]) for i=1:l_ablist]
        
    mse = sum(Flux.mse(sol_cme[i],train_sol_1[:,i]) for i=1:l_ablist)/l_ablist
    print(mse," ")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[i]), latent_size) for i=1:l_ablist]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:l_ablist])/l_ablist
    print(kl," ")

    loss = λ*mse + kl
    print(loss,"\n")
    return loss
end

function loss_func_2(p1,p2,ϵ)
    sol_cme = [sol_2(p1,p2,ab_list[i][1],ab_list[i][2],ϵ,P_0_list[i]) for i=1:l_ablist]
        
    mse = sum(Flux.mse(sol_cme[i],train_sol_2[:,i]) for i=1:l_ablist)/l_ablist
    print(mse," ")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[i]), latent_size) for i=1:l_ablist]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:l_ablist])/l_ablist
    print(kl," ")

    loss = λ*mse + kl
    print(loss,"\n")
    return loss
end

function loss_func(p1,p2,ϵ)
    loss = loss_func_1(p1,p2,ϵ) + loss_func_2(p1,p2,ϵ)
    return loss
end

λ = 50000000

#check λ if is appropriate
ϵ = zeros(latent_size)
loss_func_1(params1,params2,ϵ)
loss_func_2(params1,params2,ϵ)
loss_func(params1,params2,ϵ)
@time grads = gradient(()->loss_func(params1,params2,ϵ) , ps)

epochs_all = 0

# training
lr = 0.001;  #lr需要操作一下的
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
    solution_1 = [sol_1(params1,params2,ab_list[i][1],ab_list[i][2],ϵ,P_0_list[i]) for i=1:l_ablist]
    solution_2 = [sol_2(params1,params2,ab_list[i][1],ab_list[i][2],ϵ,P_0_list[i]) for i=1:l_ablist]

    mse_1 = sum(Flux.mse(solution_1[i],train_sol_1[:,i]) for i=1:l_ablist)/l_ablist
    mse_2 = sum(Flux.mse(solution_2[i],train_sol_2[:,i]) for i=1:l_ablist)/l_ablist
    mse = mse_1+mse_2

    if mse<mse_min[1]
        df = DataFrame( params1 = params1,params2 = vcat(params2,[0 for i=1:length(params1)-length(params2)]))
        CSV.write("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/params_tfo.csv",df)
        mse_min[1] = mse
    end

    push!(mse_list,mse)
    print(mse,"\n")
end

mse_list
mse_min 

# mse_min = [0.0001368414665310464]

using CSV,DataFrames
df = CSV.read("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/params_tfo.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

ϵ = zeros(latent_size)
solution_1 = [sol_1(params1,params2,ab_list[i][1],ab_list[i][2],ϵ,P_0_list[i]) for i=1:l_ablist]
solution_2 = [sol_2(params1,params2,ab_list[i][1],ab_list[i][2],ϵ,P_0_list[i]) for i=1:l_ablist]
mse_1 = sum(Flux.mse(solution_1[i],train_sol_1[:,i]) for i=1:l_ablist)/l_ablist
mse_2 = sum(Flux.mse(solution_2[i],train_sol_2[:,i]) for i=1:l_ablist)/l_ablist
mse = mse_1+mse_2

function plot_distribution_1(set)
    plot(0:N-1,solution_1[set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol_1[:,set],linewidth = 3,label="exact",title=join(["a,b=",ab_list[set]," var = 4800"]),line=:dash)
end

function plot_distribution_2(set)
    plot(0:N-1,solution_2[set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol_2[:,set],linewidth = 3,label="exact",title=join(["a,b=",ab_list[set]," var = 0"]),line=:dash)
end

function plot_all()
    p1 = plot_distribution_1(1)
    p2 = plot_distribution_1(2)
    p3 = plot_distribution_2(1)
    p4 = plot_distribution_2(2)
    plot(p1,p2,p3,p4,layouts=(2,2),size=(800,800))
end
plot_all()

function sol_Extenicity(τ,Attribute,a,b)
    decoder_Extenicity  = Chain(decoder_1[1],decoder_1[2],x->0.03.* x.+[i/τ for i in 1:N-1],decoder_1[4]);
    _,re2_Extenicity = Flux.destructure(decoder_Extenicity);

    function f_Extenicity!(x,p1,p2,a,b,ϵ)
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

    sol_Extenicity(p1,p2,ϵ) = nlsolve(x->f_Extenicity!(x,p1,p2,a,b,ϵ),P_0_Extenicity).zero

    P_trained_Extenicity = sol_Extenicity(params1,params2,zeros(latent_size))
    return P_trained_Extenicity
end

τ = 120
τ1 = 120
Attribute = -τ1/τ+1

# Uniform(τ1,2τ-τ1)
a = 0.0282
b = 3.46
ϵ = zeros(latent_size)
P_trained_Extenicity = sol_Extenicity(τ,Attribute,a,b)

a_list_pre = [0.0082,0.0132,0.0182,0.0232,0.0282]
b_list_pre = [1.46,1.96,2.46,2.96,3.46]
l_ablist_pre = length(a_list_pre)*length(b_list_pre)

ab_list_pre = [[a_list_pre[i],b_list_pre[j]] for i=1:length(a_list_pre) for j=1:length(b_list_pre)]

solution_list = []
for i=1:l_ablist_pre
    print(i,"\n")
    a = ab_list_pre[i][1]
    b = ab_list_pre[i][2]
    P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]

    ϵ = zeros(latent_size)
    solution = sol_Extenicity(τ,Attribute,a,b)
    push!(solution_list,solution)
end

function  plot_distribution(set)
    p=plot(0:N-1,solution_list[set],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N-1,bursty(N,ab_list_pre[set][1],ab_list_pre[set][2],τ),linewidth = 3,label="exact",line=:dash,title=join(["ab=",ab_list_pre[set]]))
end

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
    p19 = plot_distribution(19)
    p20 = plot_distribution(20)
    p21 = plot_distribution(21)
    p22 = plot_distribution(22)
    p23 = plot_distribution(23)
    p24 = plot_distribution(24)
    p25 = plot_distribution(25)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
         p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,1500),layout=(5,5))
end
plot_all()

τ1_list = [0,30,60,90,120]
Attribute_list = -τ1_list./τ.+1

a = 0.0282
b = 3.46
solution_list = []
for i=1:length(Attribute_list)
    print(i,"\n")
    Attribute = Attribute_list[i]
    P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]

    ϵ = zeros(latent_size)
    solution = sol_Extenicity(τ,Attribute,a,b)
    push!(solution_list,solution)
end
solution_list

function  plot_distribution(set)
    p=plot(0:N-1,solution_list[set],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N-1,data[:,set],linewidth = 3,label="exact",line=:dash,title=join(["τ~Uniform(",τ1_list[set],",",2τ-τ1_list[set],")"]))
end

function plot_all()
    p1 = plot_distribution(1)
    p2 = plot_distribution(2)
    p3 = plot_distribution(3)
    p4 = plot_distribution(4)
    p5 = plot_distribution(5)
    plot(p1,p2,p3,p4,p5,size=(1500,300),layout=(1,5))
end
plot_all()

# check data
# set3
# mean = 120
# α = 0.0182 β = 2.46 
# Uniform(0,240)   var = 4800 
# Uniform(30,210)  var = 2700
# Uniform(60,180)  var = 1200
# Uniform(90,150)  var = 300
# Uniform(120,120) var = 0 

# set4
# mean = 120
# α = 0.0232 β = 2.96 
# Uniform(0,240)   var = 4800 
# Uniform(30,210)  var = 2700
# Uniform(60,180)  var = 1200
# Uniform(90,150)  var = 300
# Uniform(120,120) var = 0 

check_data = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/data/check_data.csv",',')[2:end,:]

τ1_list = [0,30,60,90,120]
Attribute_list = -τ1_list./τ.+1

a = 0.0182
b = 2.46
solution_list = []
for i=1:length(Attribute_list)
    print(i,"\n")
    Attribute = Attribute_list[i]
    P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]

    ϵ = zeros(latent_size)
    solution = sol_Extenicity(τ,Attribute,a,b)
    push!(solution_list,solution)
end

function  plot_distribution(set)
    p=plot(0:N-1,solution_list[set],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N-1,check_data[:,set],linewidth = 3,label="exact",line=:dash,title=join(["τ~Uniform(",τ1_list[set],",",2τ-τ1_list[set],")"]))
end

function plot_all()
    p1 = plot_distribution(1)
    p2 = plot_distribution(2)
    p3 = plot_distribution(3)
    p4 = plot_distribution(4)
    p5 = plot_distribution(5)
    plot(p1,p2,p3,p4,p5,size=(1500,300),layout=(1,5))
end
plot_all()



