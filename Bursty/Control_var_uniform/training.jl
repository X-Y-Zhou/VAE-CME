using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

# The data is generated from 'Bursty/Control_var_uniform/SSA_uniform.jl'
# Training set
train_sol_1 = readdlm("Bursty/Control_var_uniform/data/133.csv",',')[2:end,:]
train_sol_2 = readdlm("Bursty/Control_var_uniform/data/3333.csv",',')[2:end,:]

# Testing set
train_sol_533 = readdlm("Bursty/Control_var_uniform/data/533.csv",',')[2:end,:]
train_sol_1200 = readdlm("Bursty/Control_var_uniform/data/1200.csv",',')[2:end,:]
train_sol_2133 = readdlm("Bursty/Control_var_uniform/data/2133.csv",',')[2:end,:]

# Truncation
N = 64
a = 0.0282
b = 3.46

# Model initialization
latent_size = 5;
encoder = Chain(Dense(N+1, 10,tanh),Dense(10, latent_size * 2));
decoder_1 = Chain(Dense(latent_size+1, 10,tanh),Dense(10, N),x-> 0.03*x.+[i/120 for i in 1:N],x ->relu.(x));
decoder_2 = Chain(decoder_1[1],decoder_1[2],decoder_1[3],decoder_1[4]);

params1, re1 = Flux.destructure(encoder);
params2, re2_1 = Flux.destructure(decoder_1);
      _, re2_2 = Flux.destructure(decoder_2);
ps = Flux.params(params1,params2);

#Define the CME var1 ~ 0 var1 = 133
#               var2 ~ 1 var2 = 3333
function CME_1(du, u, p, t)
    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
    z = vcat(z,0)
    NN = re2_1(p[length(params1)+1:end-latent_size])(z)

    du[1] = (-a*b/(1+b))*u[1] + NN[1]*u[2];
    for i in 2:N
            du[i] =  (-a*b/(1+b) - NN[i-1])*u[i] + NN[i]*u[i+1];
            for j in 1:i-1
                    du[i] += (a*(b/(1+b))^j/(1+b)) * u[i-j]
            end
    end
    du[N+1] = (-a*b/(1+b) - NN[N])*u[N+1];
    for j in 1:N
            du[N+1] += (a*(b/(1+b))^j/(1+b)) * u[N+1-j]
    end
end

function CME_2(du, u, p, t)
    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
    z = vcat(z,1)
    NN = re2_2(p[length(params1)+1:end-latent_size])(z)

    du[1] = (-a*b/(1+b))*u[1] + NN[1]*u[2];
    for i in 2:N
            du[i] =  (-a*b/(1+b) - NN[i-1])*u[i] + NN[i]*u[i+1];
            for j in 1:i-1
                    du[i] += (a*(b/(1+b))^j/(1+b)) * u[i-j]
            end
    end
    du[N+1] = (-a*b/(1+b) - NN[N])*u[N+1];
    for j in 1:N
            du[N+1] += (a*(b/(1+b))^j/(1+b)) * u[N+1-j]
    end
end

# Initialize the ODE solver
u0 = [1.; zeros(N)]
tf = 600;
tspan = (0, tf);
saveat = [1:5:120;140:10:600]
ϵ = zeros(latent_size)
params_all = [params1;params2;ϵ];
problem_1 = ODEProblem(CME_1, u0, tspan, params_all);
problem_2 = ODEProblem(CME_2, u0, tspan, params_all);

solution_1 = solve(problem_1,Tsit5(),u0=u0,p=params_all,saveat=saveat)
solution_2 = solve(problem_2,Tsit5(),u0=u0,p=params_all,saveat=saveat)

# Defined the CME
function loss_func_1(p1,p2,ϵ)
    params_all = [p1;p2;ϵ]
    sol_cme = solve(ODEProblem(CME_1, u0, tspan, params_all),Tsit5(),u0=u0,p=params_all,saveat=saveat)
    temp = sol_cme.u

    mse = Flux.mse(Array(sol_cme),train_sol_1[:,saveat.+1])
    print("mse:",mse," ")

    μ_logσ_list = [split_encoder_result(re1(p1)(temp[i]), latent_size) for i=1:length(saveat)]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:length(saveat)])/length(saveat)
    print(kl," ")

    loss = λ1*mse + kl
    print(loss,"\n")
    return loss
end

function loss_func_2(p1,p2,ϵ)
    params_all = [p1;p2;ϵ]
    sol_cme = solve(ODEProblem(CME_2, u0, tspan, params_all),Tsit5(),u0=u0,p=params_all,saveat=saveat)
    temp = sol_cme.u

    mse = Flux.mse(Array(sol_cme),train_sol_2[:,saveat.+1])
    print("mse:",mse," ")

    μ_logσ_list = [split_encoder_result(re1(p1)(temp[i]), latent_size) for i=1:length(saveat)]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:length(saveat)])/length(saveat)
    print(kl," ")

    loss = λ2*mse + kl
    print(loss,"\n")
    return loss
end

function loss_func(p1,p2,ϵ)
    loss = loss_func_1(p1,p2,ϵ) + loss_func_2(p1,p2,ϵ)
    return loss
end

λ1 = 3000000
λ2 = 3000000

ϵ = zeros(latent_size)
loss_func_1(params1,params2,ϵ)
loss_func_2(params1,params2,ϵ)
loss_func(params1,params2,ϵ)
grads = gradient(()->loss_func(params1,params2,ϵ),ps)

# Training process
epochs_all = 0
lr = 0.008;
opt= ADAM(lr);
epochs = 20;
epochs_all = epochs_all + epochs
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    ϵ = rand(Normal(),latent_size)
    print(epoch,"\n")
    grads = gradient(()->loss_func(params1,params2,ϵ) , ps)
    Flux.update!(opt, ps, grads)

    u0 = [1.; zeros(N)]
    tf = 600;
    tspan = (0, tf);
    saveat = 0:1:tf
    ϵ = zeros(latent_size)
    params_all = [params1;params2;ϵ];
    problem_1 = ODEProblem(CME_1, u0, tspan, params_all);
    problem_2 = ODEProblem(CME_2, u0, tspan, params_all);

    solution_1 = Array(solve(problem_1,Tsit5(),u0=u0,p=params_all,saveat=saveat))
    solution_2 = Array(solve(problem_2,Tsit5(),u0=u0,p=params_all,saveat=saveat))

    mse_1 = Flux.mse(solution_1,train_sol_1)
    mse_2 = Flux.mse(solution_2,train_sol_2)

    if mse_1+mse_2<mse_min[1]
        df = DataFrame(params1 = vcat(params1,[0 for i=1:length(params2)-length(params1)]),params2 =params2)
        CSV.write("machine-learning/ode/bursty/Controlmean_lognormal/params_trained_uniform.csv",df)
        mse_min[1] = mse_1+mse_2
    end

    push!(mse_list,mse_1+mse_2)
end

mse_list
mse_min 

mse_min = [0.00020004444384518773]

# Check
using CSV,DataFrames
df = CSV.read("Bursty/Control_var_uniform/params_trained_uniform.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

# Check var = 133
u0 = [1.; zeros(N)];
use_time=600;
time_step = 1.0;
tspan = (0.0, use_time);
params_all = [params1;params2;zeros(latent_size)];
problem = ODEProblem(CME_1, u0, tspan,params_all);
solution_1 = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)))

# Check var = 1133 
u0 = [1.; zeros(N)];
use_time=600;
time_step = 1.0; 
tspan = (0.0, use_time);
params_all = [params1;params2;zeros(latent_size)];
problem = ODEProblem(CME_2, u0, tspan,params_all);
solution_2 = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)))

mse_1 = Flux.mse(solution_1,train_sol_1)
mse_2 = Flux.mse(solution_2,train_sol_2)
mse_1+mse_2

# Check mean value 133
mean_exact = [sum([j for j=0:N].*train_sol_1[:,i]) for i=1:size(train_sol_1,2)]
mean_trained = [sum([j for j=0:N].*solution_1[:,i]) for i=1:size(solution_1,2)]
plot(mean_trained,linewidth = 3,label="VAE-CME",xlabel = "# t", ylabel = "mean value")
plot!(mean_exact,label="SSA",linewidth = 3,line=:dash,legend=:bottomright)

# Check probability distribution
function plot_distribution(time_choose)
    p=plot(0:N,solution_1[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label="SSA",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(15)
    p2 = plot_distribution(30)
    p3 = plot_distribution(45)
    p4 = plot_distribution(60)
    p5 = plot_distribution(90)
    p6 = plot_distribution(120)
    p7 = plot_distribution(150)
    p8 = plot_distribution(200)
    p9 = plot_distribution(300)
    p10 = plot_distribution(400)
    p11 = plot_distribution(500)
    p12 = plot_distribution(600)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
savefig("Bursty/Control_var_uniform/fit_var=133.svg")


# Check mean value 3333
mean_exact = [sum([j for j=0:N].*train_sol_2[:,i]) for i=1:size(train_sol_2,2)]
mean_trained = [sum([j for j=0:N].*solution_2[:,i]) for i=1:size(solution_2,2)]
plot(mean_trained,linewidth = 3,label="VAE-CME",xlabel = "# t", ylabel = "mean value")
plot!(mean_exact,label="SSA",linewidth = 3,line=:dash,legend=:bottomright)

# Check probability distribution
function plot_distribution(time_choose)
    p=plot(0:N,solution_2[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label="SSA",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(15)
    p2 = plot_distribution(30)
    p3 = plot_distribution(45)
    p4 = plot_distribution(60)
    p5 = plot_distribution(90)
    p6 = plot_distribution(120)
    p7 = plot_distribution(150)
    p8 = plot_distribution(200)
    p9 = plot_distribution(300)
    p10 = plot_distribution(400)
    p11 = plot_distribution(500)
    p12 = plot_distribution(600)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
savefig("Bursty/Control_var_uniform/fit_var=3333.svg")

# Predict probability distribution
function sol_Extenicity(τ,Attribute)
    decoder_Extenicity  = Chain(decoder_1[1],decoder_1[2],decoder_1[3],decoder_1[4]);
    _,re2_Extenicity = Flux.destructure(decoder_Extenicity);

    function CME(du, u, p, t)
        h = re1(p[1:length(params1)])(u)
        μ, logσ = split_encoder_result(h, latent_size)
        z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
        z = vcat(z,Attribute)
        NN = re2_Extenicity(p[length(params1)+1:end-latent_size])(z)
    
        du[1] = (-a*b/(1+b))*u[1] + NN[1]*u[2];
        for i in 2:N
                du[i] =  (-a*b/(1+b) - NN[i-1])*u[i] + NN[i]*u[i+1];
                for j in 1:i-1
                        du[i] += (a*(b/(1+b))^j/(1+b)) * u[i-j]
                end
        end
        du[N+1] = (-a*b/(1+b) - NN[N])*u[N+1];
        for j in 1:N
                du[N+1] += (a*(b/(1+b))^j/(1+b)) * u[N+1-j]
        end
    end

    u0 = [1.; zeros(N)];
    use_time=600;
    time_step = 1.0; 
    tspan = (0.0, use_time);
    params_all = [params1;params2;zeros(latent_size)];
    problem = ODEProblem(CME, u0, tspan,params_all);
    solution_temp = Array(solve(problem, Tsit5(), u0=u0, 
                    p=params_all, saveat=0:time_step:Int(use_time)))
    return solution_temp
end

L = 200

T1 = 20; T2 = 220
T1 = 100; T2 = 140
Attribute = -1/80 * T1 + 5/4
Attribute =  1/80 * T2 - 7/4

T1 = 40; T2 = 200; train_sol = train_sol_2133
T1 = 60; T2 = 180; train_sol = train_sol_1200
T1 = 80; T2 = 160; train_sol = train_sol_533

τ = 120
Attribute = -1/80 * T1 + 5/4
Attribute =  1/80 * T2 - 7/4
solution = sol_Extenicity(τ,Attribute)

# Check probability distribution
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label="SSA",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(15)
    p2 = plot_distribution(30)
    p3 = plot_distribution(45)
    p4 = plot_distribution(60)
    p5 = plot_distribution(90)
    p6 = plot_distribution(120)
    p7 = plot_distribution(150)
    p8 = plot_distribution(200)
    p9 = plot_distribution(300)
    p10 = plot_distribution(400)
    p11 = plot_distribution(500)
    p12 = plot_distribution(600)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
savefig("Bursty/Control_var_uniform/predict_var=533.svg")

# Relationship between T1 or T2 and Attribute
T1_list = [20,40,60,80,100]
T2_list = [220,200,180,160,140]

Attribute_list = -1/80 .* T1_list .+ 5/4
Attribute_list =  1/80 .* T2_list .- 7/4

plot(T1_list,Attribute_list,xlabel="T1/T2",ylabel="Attribute",label="T1")
scatter!(T1_list,Attribute_list,label=:false)

plot!(T2_list,Attribute_list,xlabel="T1/T2",ylabel="Attribute",label="T2")
scatter!(T2_list,Attribute_list,label=:false)
savefig("Bursty/Control_var_uniform/T1orT2_Attribute.svg")