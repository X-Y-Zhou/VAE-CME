using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

# Exact solution of Bursty Model
function bursty(N,τ,a,b)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

# Truncation
N = 64

a = 0.0282
b = 3.46
τ = 120

# Calculate the probabilities at 0~N
end_time = 600
train_sol_1 = zeros(N+1,end_time+1)
for i = 0:end_time
    a = 0.0182
    b = 3.46
    if i < τ
        train_sol_1[1:N+1,i+1] = bursty(N+1,i,a,b)
    else
        train_sol_1[1:N+1,i+1] = bursty(N+1,τ,a,b)
    end
end

end_time = 600
train_sol_2 = zeros(N+1,end_time+1)
for i = 0:end_time
    a = 0.0282
    b = 3.46
    if i < τ
        train_sol_2[1:N+1,i+1] = bursty(N+1,i,a,b)
    else
        train_sol_2[1:N+1,i+1] = bursty(N+1,τ,a,b)
    end
end

# Model initialization
latent_size = 10;
encoder = Chain(Dense(N+1, 20,tanh),Dense(20, latent_size * 2));
decoder = Chain(Dense(latent_size, 20,tanh),Dense(20, N),x-> 0.03*x.+[i/120  for i in 1:N],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

# Define the CME
function CME_1(du, u, p, t)
    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
    NN = re2(p[length(params1)+1:end-latent_size])(z)

    a = 0.0182
    b = 3.46
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
    NN = re2(p[length(params1)+1:end-latent_size])(z)

    a = 0.0282
    b = 3.46
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
tf = 600
tspan = (0, tf)
saveat = [10:10:120;140:20:600]
ϵ = zeros(latent_size)
params_all = [params1;params2;ϵ];
problem_1 = ODEProblem(CME_1, u0, tspan, params_all);
problem_2 = ODEProblem(CME_2, u0, tspan, params_all);

solution_1 = solve(problem_1,Tsit5(),u0=u0,p=params_all,saveat=saveat)
solution_2 = solve(problem_2,Tsit5(),u0=u0,p=params_all,saveat=saveat)


# Define loss function
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

λ1 = 1000000000
λ2 = 1000000000

ϵ = zeros(latent_size)
loss_func_1(params1,params2,ϵ)
loss_func_2(params1,params2,ϵ)
loss_func(params1,params2,ϵ)
grads = gradient(()->loss_func(params1,params2,ϵ),ps)

# Training process
epochs_all = 0
lr = 0.001;
opt= ADAM(lr);
epochs = 30;
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
        df = DataFrame( params1 = params1,params2 = vcat(params2,[0 for i=1:length(params1)-length(params2)]))
        CSV.write("Bursty/Control_rate/params_ode_rate2.csv",df)
        mse_min[1] = mse_1+mse_2
    end

    push!(mse_list,mse_1+mse_2)
end

mse_list
mse_min 

mse_min = [0.0003896903866546234]

# Check
using CSV,DataFrames
df = CSV.read("GitHub/VAE-CME/Bursty/Control_rate/params_ode_rate2.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

u0 = [1.; zeros(N)];
use_time=600;
time_step = 1.0; 
tspan = (0.0, use_time);
params_all = [params1;params2;zeros(latent_size)];
problem = ODEProblem(CME_1, u0, tspan,params_all);
solution_1 = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)))

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

# Check mean value
mean_exact = [sum([j for j=0:N].*train_sol[:,i]) for i=1:size(train_sol,2)]
mean_trained = [sum([j for j=0:N].*solution[:,i]) for i=1:size(solution,2)]
plot(mean_trained,linewidth = 3,label="VAE-CME",xlabel = "# t", ylabel = "mean value")
plot!(mean_exact,label="exact",linewidth = 3,line=:dash,legend=:bottomright)

# Check probability distribution
function plot_distribution(time_choose)
    p=plot(0:N,solution_1[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(27)
    p2 = plot_distribution(47)
    p3 = plot_distribution(67)
    p4 = plot_distribution(77)
    p5 = plot_distribution(87)
    p6 = plot_distribution(97)
    p7 = plot_distribution(107)
    p8 = plot_distribution(120)
    p9 = plot_distribution(200)
    p10 = plot_distribution(300)
    p11 = plot_distribution(500)
    p12 = plot_distribution(600)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()

# Check probability distribution
function plot_distribution(time_choose)
    p=plot(0:N,solution_2[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(27)
    p2 = plot_distribution(47)
    p3 = plot_distribution(67)
    p4 = plot_distribution(80)
    p5 = plot_distribution(87)
    p6 = plot_distribution(97)
    p7 = plot_distribution(107)
    p8 = plot_distribution(120)
    p9 = plot_distribution(200)
    p10 = plot_distribution(300)
    p11 = plot_distribution(500)
    p12 = plot_distribution(600)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()


function CME(du, u, p, t)
    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
    NN = re2(p[length(params1)+1:end-latent_size])(z)

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

# 0.0182~0.0282
# 0.0212,0.0252
a = 0.0252
b = 3.46
u0 = [1.; zeros(N)];
use_time=300;
time_step = 1.0; 
tspan = (0.0, use_time);
params_all = [params1;params2;zeros(latent_size)];
problem = ODEProblem(CME, u0, tspan,params_all);
solution = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)))
# sum(solution[:,end])

end_time = 300
exact_sol_extend = zeros(N+1,end_time+1)
for i = 0:end_time
    if i < τ
        exact_sol_extend[1:N+1,i+1] = bursty(N+1,i,a,b)
    else
        exact_sol_extend[1:N+1,i+1] = bursty(N+1,τ,a,b)
    end
end
exact_sol_extend

# using DataFrames,CSV
# df = DataFrame(exact_sol_extend,:auto)
# CSV.write("SSA252.csv",df)

# using DataFrames,CSV
# df = DataFrame(solution,:auto)
# CSV.write("pred252.csv",df)


function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,exact_sol_extend[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(27)
    p2 = plot_distribution(47)
    p3 = plot_distribution(67)
    p4 = plot_distribution(77)
    p5 = plot_distribution(80)
    p6 = plot_distribution(97)
    p7 = plot_distribution(107)
    p8 = plot_distribution(120)
    p9 = plot_distribution(140)
    p10 = plot_distribution(160)
    p11 = plot_distribution(180)
    p12 = plot_distribution(200)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
a = 1
# savefig("Bursty/ode_fitting.pdf")

#=
function plot_all()
    time_choose = 30
    p1=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label="VAE-CME", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label="Exact",title=join(["t=",time_choose]),line=:dash)
    
    time_choose = 45
    p2=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)
    
    time_choose = 60
    p3=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 75
    p4=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 90
    p5=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false, ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 105
    p6=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 120
    p7=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)
    
    time_choose = 150
    p8=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 200
    p9=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 300
    p10=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 400
    p11=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 600
    p12=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
savefig("Bursty/Bursty_fitting.svg")
=#