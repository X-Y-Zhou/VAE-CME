using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

function bursty(N,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

# truncation
N = 64

a = 0.0282
b = 3.46
τ = 120
P_exat = bursty(N+1,τ)

# calculate the probabilities at 0~N
end_time = 1200
train_sol_120 = zeros(N+1,end_time+1)
for i = 0:end_time
    if i < 120
        train_sol_120[1:N+1,i+1] = bursty(N+1,i)
    else
        train_sol_120[1:N+1,i+1] = bursty(N+1,120)
    end
end

end_time = 1200
train_sol_30 = zeros(N+1,end_time+1)
for i = 0:end_time
    if i < 30
        train_sol_30[1:N+1,i+1] = bursty(N+1,i)
    else
        train_sol_30[1:N+1,i+1] = bursty(N+1,30)
    end
end

# model initialization
latent_size = 5;
encoder = Chain(Dense(N+1, 10,tanh),Dense(10, latent_size * 2));
decoder_120 = Chain(Dense(latent_size+1, 10,tanh),Dense(10, N),x-> 0.03*x.+[i/120 for i in 1:N],x ->relu.(x));
decoder_30  = Chain(decoder_120[1],decoder_120[2],x-> 0.03*x.+[i/30 for i in 1:N],decoder_120[4]);

params1, re1 = Flux.destructure(encoder);
params2, re2_120 = Flux.destructure(decoder_120);
      _, re2_30 = Flux.destructure(decoder_30);
ps = Flux.params(params1,params2);

# define the CME
function CME_120(du, u, p, t)
    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
    z = vcat(z,1)
    NN = re2_120(p[length(params1)+1:end-latent_size])(z)

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

function CME_30(du, u, p, t)
    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
    z = vcat(z,0)
    NN = re2_30(p[length(params1)+1:end-latent_size])(z)

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

# initialize the ODE solver
u0 = [1.; zeros(N)]
tf = 1200; #end time
tspan = (0, tf);
saveat = [10:5:120;140:20:1200]
ϵ = zeros(latent_size)
params_all = [params1;params2;ϵ];
problem_120 = ODEProblem(CME_120, u0, tspan, params_all);
problem_30 = ODEProblem(CME_30, u0, tspan, params_all);

solution_120 = solve(problem_120,Tsit5(),u0=u0,p=params_all,saveat=saveat)
solution_30 = solve(problem_30,Tsit5(),u0=u0,p=params_all,saveat=saveat)

function loss_func_120(p1,p2,ϵ)
    params_all = [p1;p2;ϵ]
    sol_cme = solve(ODEProblem(CME_120, u0, tspan, params_all),Tsit5(),u0=u0,p=params_all,saveat=saveat)
    temp = sol_cme.u

    μ_logσ_list = [split_encoder_result(re1(p1)(temp[i]), latent_size) for i=1:length(saveat)]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:length(saveat)])/length(saveat)
    print(kl," ")

    reg_zero = Flux.mse(Array(sol_cme),train_sol_120[:,saveat.+1])
    print(reg_zero," ")

    loss = kl + λ1*reg_zero

    print(loss,"\n")
    return loss
end

function loss_func_30(p1,p2,ϵ)
    params_all = [p1;p2;ϵ]
    sol_cme = solve(ODEProblem(CME_30, u0, tspan, params_all),Tsit5(),u0=u0,p=params_all,saveat=saveat)
    temp = sol_cme.u

    μ_logσ_list = [split_encoder_result(re1(p1)(temp[i]), latent_size) for i=1:length(saveat)]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:length(saveat)])/length(saveat)
    print(kl," ")

    reg_zero = Flux.mse(Array(sol_cme),train_sol_30[:,saveat.+1])
    print(reg_zero," ")

    loss = kl + λ2*reg_zero

    print(loss,"\n")
    return loss
end

function loss_func(p1,p2,ϵ)
    loss = loss_func_120(p1,p2,ϵ) + loss_func_30(p1,p2,ϵ)
    return loss
end

λ1 = 5000000
λ2 = 100000

ϵ = zeros(latent_size)
loss_func_120(params1,params2,ϵ)
loss_func_30(params1,params2,ϵ)
loss_func(params1,params2,ϵ)
grads = gradient(()->loss_func(params1,params2,ϵ),ps)

epochs_all = 0

lr = 0.002;
opt= ADAM(lr);
epochs = 10;
epochs_all = epochs_all + epochs
print("learning rate = ",lr)
@time for epoch in 1:epochs
    ϵ = rand(Normal(),latent_size)
    print(epoch,"\n")
    grads = gradient(()->loss_func(params1,params2,ϵ) , ps)
    Flux.update!(opt, ps, grads)

    solution_time_points_120 = Array(solve(ODEProblem(CME_120, u0, tspan, [params1;params2;zeros(latent_size)]), 
                                    Tsit5(), p=[params1;params2;zeros(latent_size)],u0=u0, saveat=saveat))
    train_timepoints_120 = train_sol_120[:,saveat.+1]
    mse_120 = Flux.mse(solution_time_points_120,train_timepoints_120)

    solution_time_points_30 = Array(solve(ODEProblem(CME_30, u0, tspan, [params1;params2;zeros(latent_size)]), 
                Tsit5(), p=[params1;params2;zeros(latent_size)],u0=u0, saveat=saveat))
    train_timepoints_30 = train_sol_30[:,saveat.+1]
    mse_30 = Flux.mse(solution_time_points_30,train_timepoints_30)
    print(mse_120," ",mse_30,"\n")
    print(mse_120+mse_30,"\n")
end


# check τ = 120
u0 = [1.; zeros(N)];
use_time=1200;
time_step = 1.0;
tspan = (0.0, use_time);
params_all = [params1;params2;zeros(latent_size)];
problem = ODEProblem(CME_120, u0, tspan,params_all);
solution = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)))

#check mean
mean_exact = [sum([j for j=0:N].*train_sol_120[:,i]) for i=1:size(train_sol_120,2)]
mean_trained = [sum([j for j=0:N].*solution[:,i]) for i=1:size(solution,2)]
plot(mean_trained,linewidth = 3,label="VAE-CME",xlabel = "# t", ylabel = "mean value")
plot!(mean_exact,label="exact",linewidth = 3,line=:dash,legend=:bottomright)

#check probabilities
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol_120[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
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
    p12 = plot_distribution(800)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
savefig("Bursty/figs/tau=120.pdf")

# check τ = 30
u0 = [1.; zeros(N)];
use_time=1200;
time_step = 1.0; 
tspan = (0.0, use_time);
params_all = [params1;params2;zeros(latent_size)];
problem = ODEProblem(CME_30, u0, tspan,params_all);
solution = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)))

#check mean
mean_exact = [sum([j for j=0:N].*train_sol_30[:,i]) for i=1:size(train_sol_120,2)]
mean_trained = [sum([j for j=0:N].*solution[:,i]) for i=1:size(solution,2)]
plot(mean_trained,linewidth = 3,label="VAE-CME",xlabel = "# t", ylabel = "mean value")
plot!(mean_exact,label="exact",linewidth = 3,line=:dash,legend=:bottomright)

#check probabilities
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol_30[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(5)
    p2 = plot_distribution(10)
    p3 = plot_distribution(15)
    p4 = plot_distribution(20)
    p5 = plot_distribution(25)
    p6 = plot_distribution(30)
    p7 = plot_distribution(40)
    p8 = plot_distribution(50)
    p9 = plot_distribution(60)
    p10 = plot_distribution(90)
    p11 = plot_distribution(120)
    p12 = plot_distribution(150)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
savefig("Bursty/figs/tau=30.pdf")

#read params
using CSV,DataFrames
df = CSV.read("Bursty/Control_mean/params_ode_control_mean.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

#write params
using DataFrames,CSV
params1
params2
df = DataFrame(params1 = vcat(params1,[0 for i=1:length(params2)-length(params1)]),params2 =params2)
CSV.write("machine-learning/ode/bursty/after20230317/params_trained_origin2.csv",df)


function sol_Extenicity(τ,Attribute)
    decoder_Extenicity  = Chain(decoder_120[1],decoder_120[2],x-> 0.03*x.+[i/τ for i in 1:N],decoder_120[4]);
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
    use_time=120;
    time_step = 1.0; 
    tspan = (0.0, use_time);
    params_all = [params1;params2;zeros(latent_size)];
    problem = ODEProblem(CME, u0, tspan,params_all);
    solution_temp = Array(solve(problem, Tsit5(), u0=u0, 
                    p=params_all, saveat=0:time_step:Int(use_time)))
    return solution_temp
end

τ = 40
Attribute = -40/τ+4/3

solution = sol_Extenicity(τ,Attribute)

end_time = 120
train_sol = zeros(N+1,end_time+1)
for i = 0:end_time
    if i < τ
        train_sol[1:N+1,i+1] = bursty(N+1,i)
    else
        train_sol[1:N+1,i+1] = bursty(N+1,τ)
    end
end

#check probabilities
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(17)
    p2 = plot_distribution(27)
    p3 = plot_distribution(37)
    p4 = plot_distribution(47)
    p5 = plot_distribution(57)
    p6 = plot_distribution(67)
    p7 = plot_distribution(77)
    p8 = plot_distribution(87)
    p9 = plot_distribution(97)
    p10 = plot_distribution(105)
    p11 = plot_distribution(107)
    p12 = plot_distribution(120)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
savefig("Bursty/figs/pre_tau=$τ.pdf")

τ_list = [1/30,1/40,1/60,1/100,1/120]
Attribute_list = -40. *τ_list.+4/3
plot(τ_list,Attribute_list,xlabel="1/τ",ylabel="Attribute",legend=:false)
scatter!(τ_list,Attribute_list,legend=:false)
savefig("Bursty/Control_mean/τ_Attribute.pdf")