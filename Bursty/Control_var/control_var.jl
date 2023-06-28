using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

# the data is generated from 'variable_tau_theory/SSA_car.jl'
# training set
train_sol_2 = readdlm("Bursty/Control_var/data/4266.csv",',')[2:end,:]
train_sol_2 = readdlm("Bursty/Control_var/data/11266.csv",',')[2:end,:]

# testing set
train_sol_5400 = readdlm("Bursty/Control_var/data/5400.csv",',')[2:end,:]
train_sol_7350 = readdlm("Bursty/Control_var/data/7350.csv",',')[2:end,:]
train_sol_9600 = readdlm("Bursty/Control_var/data/9600.csv",',')[2:end,:]

# truncation
N = 64

a = 0.0282
b = 3.46
τ = 120

# model initialization
latent_size = 5;
encoder = Chain(Dense(N+1, 10,tanh),Dense(10, latent_size * 2));
decoder_1 = Chain(Dense(latent_size+1, 10,tanh),Dense(10, N),x-> 0.03*x.+[i/120 for i in 1:N],x ->relu.(x));
decoder_2 = Chain(decoder_1[1],decoder_1[2],decoder_1[3],decoder_1[4]);

params1, re1 = Flux.destructure(encoder);
params2, re2_1 = Flux.destructure(decoder_1);
      _, re2_2 = Flux.destructure(decoder_2);
ps = Flux.params(params1,params2);

#CME var1 ~ 0 var = 4266
#    var2 ~ 1 var = 11266
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

# initialize the ODE solver
u0 = [1.; zeros(N)]
tf = 800; #end time
tspan = (0, tf);
saveat = [1:5:120;140:20:800]
ϵ = zeros(latent_size)
params_all = [params1;params2;ϵ];
problem_1 = ODEProblem(CME_1, u0, tspan, params_all);
problem_2 = ODEProblem(CME_2, u0, tspan, params_all);

solution_1 = solve(problem_1,Tsit5(),u0=u0,p=params_all,saveat=saveat)
solution_2 = solve(problem_2,Tsit5(),u0=u0,p=params_all,saveat=saveat)

function loss_func_1(p1,p2,ϵ)
    params_all = [p1;p2;ϵ]
    sol_cme = solve(ODEProblem(CME_1, u0, tspan, params_all),Tsit5(),u0=u0,p=params_all,saveat=saveat)
    temp = sol_cme.u

    μ_logσ_list = [split_encoder_result(re1(p1)(temp[i]), latent_size) for i=1:length(saveat)]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:length(saveat)])/length(saveat)
    print(kl," ")

    reg_zero = Flux.mse(Array(sol_cme),train_sol_2[:,saveat.+1])
    print(reg_zero," ")

    loss = kl + λ1*reg_zero

    print(loss,"\n")
    return loss
end

function loss_func_2(p1,p2,ϵ)
    params_all = [p1;p2;ϵ]
    sol_cme = solve(ODEProblem(CME_2, u0, tspan, params_all),Tsit5(),u0=u0,p=params_all,saveat=saveat)
    temp = sol_cme.u

    μ_logσ_list = [split_encoder_result(re1(p1)(temp[i]), latent_size) for i=1:length(saveat)]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:length(saveat)])/length(saveat)
    print(kl," ")

    reg_zero = Flux.mse(Array(sol_cme),train_sol_2[:,saveat.+1])
    print(reg_zero," ")

    loss = kl + λ2*reg_zero

    print(loss,"\n")
    return loss
end

function loss_func(p1,p2,ϵ)
    loss = loss_func_1(p1,p2,ϵ) + loss_func_2(p1,p2,ϵ)
    return loss
end

λ1 = 30000000
λ2 = 10000000

ϵ = zeros(latent_size)
loss_func_1(params1,params2,ϵ)
loss_func_2(params1,params2,ϵ)
loss_func(params1,params2,ϵ)
grads = gradient(()->loss_func(params1,params2,ϵ),ps)

epochs_all = 0

lr = 0.0006;
opt= ADAM(lr);
epochs = 10;
epochs_all = epochs_all + epochs
print("learning rate = ",lr)
@time for epoch in 1:epochs
    ϵ = rand(Normal(),latent_size)
    print(epoch,"\n")
    grads = gradient(()->loss_func(params1,params2,ϵ) , ps)
    Flux.update!(opt, ps, grads)

    solution_time_points_1 = Array(solve(ODEProblem(CME_1, u0, tspan, [params1;params2;zeros(latent_size)]), 
                                    Tsit5(), p=[params1;params2;zeros(latent_size)],u0=u0, saveat=saveat))
    train_timepoints_1 = train_sol_2[:,saveat.+1]
    mse_1 = Flux.mse(solution_time_points_1,train_timepoints_1)

    solution_time_points_2 = Array(solve(ODEProblem(CME_2, u0, tspan, [params1;params2;zeros(latent_size)]), 
                Tsit5(), p=[params1;params2;zeros(latent_size)],u0=u0, saveat=saveat))
    train_timepoints_2 = train_sol_2[:,saveat.+1]
    mse_2 = Flux.mse(solution_time_points_2,train_timepoints_2)
    print(mse_1," ",mse_2,"\n")
    print(mse_1+mse_2,"\n")
end


# check var = 4266 
u0 = [1.; zeros(N)];
use_time=800;
time_step = 1.0;
tspan = (0.0, use_time);
params_all = [params1;params2;zeros(latent_size)];
problem = ODEProblem(CME_1, u0, tspan,params_all);
solution = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)))

#check mean
mean_exact = [sum([j for j=0:N].*train_sol_2[:,i]) for i=1:size(train_sol_2,2)]
mean_trained = [sum([j for j=0:N].*solution[:,i]) for i=1:size(solution,2)]
plot(mean_trained,linewidth = 3,label="VAE-CME",xlabel = "# t", ylabel = "mean value")
plot!(mean_exact,label="exact",linewidth = 3,line=:dash,legend=:bottomright)

#check probabilities
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
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

savefig("Bursty/Control_var/fit_var=4266.pdf")

# check var = 11266 
u0 = [1.; zeros(N)];
use_time=800;
time_step = 1.0; 
tspan = (0.0, use_time);
params_all = [params1;params2;zeros(latent_size)];
problem = ODEProblem(CME_2, u0, tspan,params_all);
solution = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)))

#check mean
mean_exact = [sum([j for j=0:N].*train_sol_2[:,i]) for i=1:size(train_sol_2,2)]
mean_trained = [sum([j for j=0:N].*solution[:,i]) for i=1:size(solution,2)]
plot(mean_trained,linewidth = 3,label="VAE-CME",xlabel = "# t", ylabel = "mean value")
plot!(mean_exact,label="exact",linewidth = 3,line=:dash,legend=:bottomright)

#check probabilities
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
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
savefig("Bursty/Control_var/fit_var=11266.pdf")

#read params
using CSV,DataFrames
df = CSV.read("Bursty/Control_var/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

#write params
using DataFrames,CSV
params1
params2
df = DataFrame(params1 = vcat(params1,[0 for i=1:length(params2)-length(params1)]),params2 =params2)
CSV.write("machine-learning/ode/car_problem/params_trained2.csv",df)


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
    use_time=800;
    time_step = 1.0; 
    tspan = (0.0, use_time);
    params_all = [params1;params2;zeros(latent_size)];
    problem = ODEProblem(CME, u0, tspan,params_all);
    solution_temp = Array(solve(problem, Tsit5(), u0=u0, 
                    p=params_all, saveat=0:time_step:Int(use_time)))
    return solution_temp
end

L = 200

T1 = L/6; T2 = 1.25L
Attribute = -6/L * T1 +2
Attribute = 4/L* T2-4

T1 = 3L/10; T2 = 1.05L; train_sol = train_sol_5400
T1 = L/4; T2 = 1.125L;  train_sol = train_sol_7350
T1 = L/5; T2 = 1.2L;    train_sol = train_sol_9600

Attribute = -6/L*T1 +2
Attribute = 4/L*T2-4

solution = sol_Extenicity(τ,Attribute)

function plot_all()
    time_choose = 30
    p1=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label="VAE-CME", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    
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

    time_choose = 800
    p12=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all();
savefig("Bursty/Control_var/predict_var=9600.svg")
savefig("Statement/Figs/predict_var=9600.pdf")


#check probabilities
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(17)
    p2 = plot_distribution(37)
    p3 = plot_distribution(67)
    p4 = plot_distribution(97)
    p5 = plot_distribution(117)
    p6 = plot_distribution(157)
    p7 = plot_distribution(207)
    p8 = plot_distribution(257)
    p9 = plot_distribution(300)
    p10 = plot_distribution(400)
    p11 = plot_distribution(600)
    p12 = plot_distribution(800)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
savefig("Bursty/Control_var/predict_var=9600.pdf")

T1_list = [L/3,3L/10,L/4,L/5,L/6]
T2_list = [L,1.05L,1.125L,1.2L,1.25L]

Attribute_list = -6/L .* T1_list.+2
Attribute_list =  4/L .* T2_list.-4

plot(T1_list,Attribute_list,xlabel="T1/T2",ylabel="Attribute",label="T1")
scatter!(T1_list,Attribute_list,label=:false)

plot!(T2_list,Attribute_list,xlabel="T1/T2",ylabel="Attribute",label="T2")
scatter!(T2_list,Attribute_list,label=:false)

savefig("Bursty/Control_var/T1orT2_Attribute.svg")
savefig("Statement/Figs/T1orT2_Attribute.pdf")


using CSV,DataFrames
df = CSV.read("Bursty/Control_var/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

# check var = 4266 
u0 = [1.; zeros(N)];
use_time=800;
time_step = 1.0;
tspan = (0.0, use_time);
params_all = [params1;params2;zeros(latent_size)];
problem = ODEProblem(CME_1, u0, tspan,params_all);
solution = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)))

function plot_all()
    time_choose = 30
    p1=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label="VAE-CME", ylabel = "\n Probability")
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    
    time_choose = 45
    p2=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)
    
    time_choose = 60
    p3=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 75
    p4=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 90
    p5=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false, ylabel = "\n Probability")
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 105
    p6=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 120
    p7=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)
    
    time_choose = 150
    p8=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 200
    p9=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 300
    p10=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 400
    p11=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 800
    p12=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
savefig("Bursty/Control_var/fit_var=4266.svg")
savefig("Statement/Figs/fit_var=4266.pdf")

# check var = 11266 
u0 = [1.; zeros(N)];
use_time=800;
time_step = 1.0; 
tspan = (0.0, use_time);
params_all = [params1;params2;zeros(latent_size)];
problem = ODEProblem(CME_2, u0, tspan,params_all);
solution = Array(solve(problem, Tsit5(), u0=u0, 
                 p=params_all, saveat=0:time_step:Int(use_time)))

function plot_all()
    time_choose = 30
    p1=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label="VAE-CME", ylabel = "\n Probability")
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    
    time_choose = 45
    p2=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)
    
    time_choose = 60
    p3=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 75
    p4=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose+1]),line=:dash)

    time_choose = 90
    p5=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false, ylabel = "\n Probability")
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 105
    p6=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 120
    p7=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)
    
    time_choose = 150
    p8=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 200
    p9=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 300
    p10=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 400
    p11=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    time_choose = 800
    p12=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose]),line=:dash)

    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
savefig("Bursty/Control_var/fit_var=11266.svg")
savefig("Statement/Figs/fit_var=11266.pdf")