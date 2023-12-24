using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

# training data
# ρ = 0.5
# mean = e^4
# LogNormal(4,sqrt(0)) [0]  # var = 0
# LogNormal(3,sqrt(2))      # var = 19045
# LogNormal(2,sqrt(4))      # var = 159773
# LogNormal(1,sqrt(6))      # var = 1199623
# LogNormal(0,sqrt(8)) [1]  # var = 8883129

# training data
# ρ = 2.0
# mean = e^4
# LogNormal(4,sqrt(0)) [0]  # var = 0
# LogNormal(3,sqrt(2))      # var = 19045
# LogNormal(2,sqrt(4))      # var = 159773
# LogNormal(1,sqrt(6))      # var = 1199623
# LogNormal(0,sqrt(8)) [1]  # var = 8883129

N = 200
τ = exp(4)
data1 = readdlm("Birth-Death/Inference2/data/ρ=0.5.csv",',')[2:end,:]
data2 = readdlm("Birth-Death/Inference2/data/ρ=2.0.csv",',')[2:end,:]
train_sol_1 = [data1[:,1] data2[:,1]] # var = 0 ρ = 0.5,2.0
train_sol_2 = [data1[:,5] data2[:,5]] # var = 8883129 ρ = 0.5,2.0

train_sol_1 = data1[:,1] # var = 0 ρ = 0.5
train_sol_2 = data1[:,5] # var = 8883129 ρ = 0.5
train_sol_3 = data1[:,3] # var = 8883129 ρ = 0.5

ρ_list = [0.5]
l_ρ_list = length(ρ_list)

# model initialization
latent_size = 2;
encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
decoder_1 = Chain(Dense(latent_size+1, 10),Dense(10 , N-1),x->0.03.* x.+[i/τ  for i in 1:N-1],x ->relu.(x));
decoder_2  = Chain(decoder_1[1],decoder_1[2],decoder_1[3],decoder_1[4]);
decoder_3  = Chain(decoder_1[1],decoder_1[2],decoder_1[3],decoder_1[4]);

params1, re1 = Flux.destructure(encoder);
params2, re2_1 = Flux.destructure(decoder_1);
      _, re2_2 = Flux.destructure(decoder_2);
      _, re2_3 = Flux.destructure(decoder_3);
ps = Flux.params(params1,params2);

params1
params2

#CME 0~0
function f1!(x,p1,p2,ρ,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,0)
    NN = re2_1(p2)(z)
    return vcat(-ρ*x[1] + NN[1]*x[2],
                [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
                sum(x)-1)
end

# 8883129~1
function f2!(x,p1,p2,ρ,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,1)
    NN = re2_2(p2)(z)
    return vcat(-ρ*x[1] + NN[1]*x[2],
                [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
                sum(x)-1)
end

function f3!(x,p1,p2,ρ,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,0.5)
    NN = re2_3(p2)(z)
    return vcat(-ρ*x[1] + NN[1]*x[2],
                [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
                sum(x)-1)
end

#solve P
P_0_list = [[pdf(Poisson(ρ_list[i]*τ),j) for j=0:N-1] for i=1:l_ρ_list]

ϵ = zeros(latent_size)
sol_1(p1,p2,ρ,ϵ,P0) = nlsolve(x->f1!(x,p1,p2,ρ,ϵ),P0).zero
sol_2(p1,p2,ρ,ϵ,P0) = nlsolve(x->f2!(x,p1,p2,ρ,ϵ),P0).zero
sol_3(p1,p2,ρ,ϵ,P0) = nlsolve(x->f3!(x,p1,p2,ρ,ϵ),P0).zero

function loss_func_1(p1,p2,ϵ)
    sol_cme = [sol_1(p1,p2,ρ_list[i],ϵ,P_0_list[i]) for i=1:l_ρ_list]
        
    mse = sum(Flux.mse(sol_cme[i],train_sol_1[:,i]) for i=1:l_ρ_list)/l_ρ_list
    print(mse," ")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[i]), latent_size) for i=1:l_ρ_list]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:l_ρ_list])/l_ρ_list
    print(kl," ")

    loss = λ*mse + kl
    print(loss,"\n")
    return loss
end

function loss_func_2(p1,p2,ϵ)
    sol_cme = [sol_2(p1,p2,ρ_list[i],ϵ,P_0_list[i]) for i=1:l_ρ_list]
        
    mse = sum(Flux.mse(sol_cme[i],train_sol_2[:,i]) for i=1:l_ρ_list)/l_ρ_list
    print(mse," ")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[i]), latent_size) for i=1:l_ρ_list]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:l_ρ_list])/l_ρ_list
    print(kl," ")

    loss = λ*mse + kl
    print(loss,"\n")
    return loss
end

function loss_func_3(p1,p2,ϵ)
    sol_cme = [sol_3(p1,p2,ρ_list[i],ϵ,P_0_list[i]) for i=1:l_ρ_list]
        
    mse = sum(Flux.mse(sol_cme[i],train_sol_3[:,i]) for i=1:l_ρ_list)/l_ρ_list
    print(mse," ")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[i]), latent_size) for i=1:l_ρ_list]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:l_ρ_list])/l_ρ_list
    print(kl," ")

    loss = λ*mse + kl
    print(loss,"\n")
    return loss
end

function loss_func(p1,p2,ϵ)
    loss = loss_func_1(p1,p2,ϵ) + loss_func_2(p1,p2,ϵ) + loss_func_3(p1,p2,ϵ)
    return loss
end

λ = 1000000

#check λ if is appropriate
ϵ = zeros(latent_size)
loss_func_1(params1,params2,ϵ)
loss_func_2(params1,params2,ϵ)
loss_func_3(params1,params2,ϵ)
loss_func(params1,params2,ϵ)
@time grads = gradient(()->loss_func(params1,params2,ϵ) , ps)

epochs_all = 0

# training
lr = 0.025;  #lr需要操作一下的
lr_list = [0.025,0.01,0.008,0.006,0.004,0.002,0.001]
lr_list = [0.025]

for lr in lr_list
using CSV,DataFrames
df = CSV.read("Birth-Death/Inference2/params_training-2.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

opt= ADAM(lr);
epochs = 30
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    ϵ = rand(Normal(),latent_size)
    print(epoch,"\n")
    grads = gradient(()->loss_func(params1,params2,ϵ) , ps)
    Flux.update!(opt, ps, grads)

    ϵ = zeros(latent_size)
    solution_1 = [sol_1(params1,params2,ρ_list[i],ϵ,P_0_list[i]) for i=1:l_ρ_list]
    solution_2 = [sol_2(params1,params2,ρ_list[i],ϵ,P_0_list[i]) for i=1:l_ρ_list]
    solution_3 = [sol_3(params1,params2,ρ_list[i],ϵ,P_0_list[i]) for i=1:l_ρ_list]

    mse_1 = sum(Flux.mse(solution_1[i],train_sol_1[:,i]) for i=1:l_ρ_list)/l_ρ_list
    mse_2 = sum(Flux.mse(solution_2[i],train_sol_2[:,i]) for i=1:l_ρ_list)/l_ρ_list
    mse_3 = sum(Flux.mse(solution_3[i],train_sol_3[:,i]) for i=1:l_ρ_list)/l_ρ_list
    mse = mse_1+mse_2+mse_3

    if mse<mse_min[1]
        df = DataFrame(params1 = vcat(params1,[0 for i=1:length(params2)-length(params1)]),params2 = params2)
        CSV.write("Birth-Death/Inference2/params_training-2.csv",df)
        mse_min[1] = mse
    end

    push!(mse_list,mse)
    print(mse,"\n")
end
end

mse_list
mse_min 

mse_min = [0.00016619896562221242]

using CSV,DataFrames
df = CSV.read("Birth-Death/Inference2/params_training-2.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

ϵ = zeros(latent_size)
solution_1 = [sol_1(params1,params2,ρ_list[i],ϵ,P_0_list[i]) for i=1:l_ρ_list]
solution_2 = [sol_2(params1,params2,ρ_list[i],ϵ,P_0_list[i]) for i=1:l_ρ_list]
solution_3 = [sol_3(params1,params2,ρ_list[i],ϵ,P_0_list[i]) for i=1:l_ρ_list]
mse_1 = sum(Flux.mse(solution_1[i],train_sol_1[:,i]) for i=1:l_ρ_list)/l_ρ_list
mse_2 = sum(Flux.mse(solution_2[i],train_sol_2[:,i]) for i=1:l_ρ_list)/l_ρ_list
mse_3 = sum(Flux.mse(solution_3[i],train_sol_3[:,i]) for i=1:l_ρ_list)/l_ρ_list
mse = mse_1+mse_2+mse_3

set = 1
train_sol_1[:,set]

function plot_distribution_1(set)
    plot(0:N-1,solution_1[set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol_1[:,set],linewidth = 3,label="exact",title=join(["ρ=",ρ_list[set]," var = 0"]),line=:dash)
end

function plot_distribution_2(set)
    plot(0:N-1,solution_2[set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol_2[:,set],linewidth = 3,label="exact",title=join(["ρ=",ρ_list[set]," var = max"]),line=:dash)
end

function plot_distribution_3(set)
    plot(0:N-1,solution_3[set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol_3[:,set],linewidth = 3,label="exact",title=join(["ρ=",ρ_list[set]," var = max"]),line=:dash)
end

function plot_all()
    p1 = plot_distribution_1(1)
    p2 = plot_distribution_2(1)
    # p2 = plot_distribution_1(2)
    p3 = plot_distribution_3(1)
    # p4 = plot_distribution_2(2)
    plot(p1,p2,p3,layouts=(1,3),size=(1200,400))
end
plot_all()

function sol_Extenicity(τ,Attribute,ρ)
    decoder_Extenicity  = Chain(decoder_1[1],decoder_1[2],x->0.03.* x.+[i/τ for i in 1:N-1],decoder_1[4]);
    _,re2_Extenicity = Flux.destructure(decoder_Extenicity);

    function f_Extenicity!(x,p1,p2,ρ,ϵ)
        h = re1(p1)(x)
        μ, logσ = split_encoder_result(h, latent_size)
        z = reparameterize.(μ, logσ, ϵ)
        z = vcat(z,Attribute)
        NN = re2_Extenicity(p2)(z)
        return vcat(-ρ*x[1] + NN[1]*x[2],
                [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
                sum(x)-1)
    end

    P_0_distribution_Extenicity = Poisson(ρ*τ);
    P_0_Extenicity = [pdf(P_0_distribution_Extenicity,i) for i=0:N-1]

    sol_Extenicity(p1,p2,ϵ) = nlsolve(x->f_Extenicity!(x,p1,p2,ρ,ϵ),P_0_Extenicity).zero

    P_trained_Extenicity = sol_Extenicity(params1,params2,zeros(latent_size))
    return P_trained_Extenicity
end

μ = 0
Attribute = -μ/4+1

# Uniform(τ1,2τ-τ1)
ρ = 0.5
ϵ = zeros(latent_size)
P_trained_Extenicity = sol_Extenicity(τ,Attribute,ρ)

p=plot(0:N-1,P_trained_Extenicity,linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,data1[:,1],linewidth = 3,label="exact",line=:dash,)



a_list_pre = [0.0082,0.0132,0.0182,0.0232,0.0282]
b_list_pre = [1.46,1.96,2.46,2.96,3.46]
l_ρ_list_pre = length(a_list_pre)*length(b_list_pre)

ab_list_pre = [[a_list_pre[i],b_list_pre[j]] for i=1:length(a_list_pre) for j=1:length(b_list_pre)]

solution_list = []
for i=1:l_ρ_list_pre
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

# set5
# mean = 120
# α = 0.0182 β = 2.96 
# Uniform(0,240)   var = 4800 
# Uniform(30,210)  var = 2700
# Uniform(60,180)  var = 1200
# Uniform(90,150)  var = 300
# Uniform(120,120) var = 0 

check_data = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/data/check_data.csv",',')[2:end,:]
check_data[:,1:5]
check_data[:,6:10]

τ1_list = [0,30,60,90,120]
Attribute_list = -τ1_list./τ.+1

a = 0.0182
b = 2.96
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
    plot!(0:N-1,check_data[:,set+10],linewidth = 3,label="exact",line=:dash,title=join(["τ~Uniform(",τ1_list[set],",",2τ-τ1_list[set],")"]))
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



