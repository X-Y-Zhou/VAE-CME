using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../utils.jl")

# Load training data
data = readdlm("On-Off/delay_telegraph_plateau1.txt", ':')
N = Int(maximum(data))

train_sol = zeros(N+1,size(data,1))
for i =1:size(data,1)
    probability = convert_histo(vec(data[i,:]))[2]
    if length(probability)<N+1
        train_sol[1:length(probability),i] = probability
    else
        train_sol[1:N+1,i] = probability[1:N+1]
    end
end

sigma_on=1.0;
sigma_off=1.0;
rho_on=20.0; 
rho_off=0.0;
gamma=0.0;

# Model initialization
latent_size = 5;
encoder = Chain(Dense(2*(N+1), 10,tanh),Dense(10, latent_size * 2));
decoder_1 = Chain(Dense(latent_size, 10,tanh),Dense(10, N),x-> x.+[i/1 for i in 1:N],x ->relu.(x));
decoder_2  = Chain(Dense(latent_size, 10,tanh),Dense(10, N),x-> x.+[i/1 for i in 1:N],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2_1, re2_1 = Flux.destructure(decoder_1);
params2_2, re2_2 = Flux.destructure(decoder_2);
ps = Flux.params(params1,params2_1,params2_2);

# Define the CME
function CME(du, u, p, t)
    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])

    NN1 = re2_1(p[length(params1)+1:length(params1)+length(params2_1)])(z)
    NN2 = re2_2(p[length(params1)+length(params2_1)+1:end-latent_size])(z)

    du[1] = (-sigma_on-rho_off)*u[1] + (-gamma+NN1[1])*u[2] + sigma_off*u[N+2]
    for i in 2:N
        du[i] = rho_off*u[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*u[i] + (-i*gamma+NN1[i])*u[i+1] + sigma_off*u[i+N+1];
    end
    du[N+1] = rho_off*u[N] + (-sigma_on-rho_off+N*gamma-NN1[N])*u[N+1] + sigma_off*u[2*N+2];

    du[N+2] = sigma_on*u[1] + (-sigma_off-rho_on)*u[N+2] + (-gamma+NN2[1])*u[N+3]
    for i in (N+3):(2*N+1)
        du[i] = sigma_on*u[i-N-1] + rho_on*u[i-1] + (-sigma_off-rho_on+(i-N-2)*gamma -NN2[i-N-2])*u[i] + (-(i-N-1)*gamma+NN2[i-N-1])*u[i+1]
    end
    du[2*N+2] = sigma_on*u[N+1] +rho_on*u[2*N+1]+(-sigma_off-rho_on+N*gamma -NN2[N])*u[2*N+2]
end

# Initialize the ODE solver
u0 = [1.; zeros(2*N+1)]
tf = 10;
tspan = (0, tf);
saveat = 0:0.1:10
ϵ = zeros(latent_size)
params_all = [params1;params2_1;params2_2;ϵ];
problem = ODEProblem(CME, u0, tspan, params_all);
solution = Array(solve(problem,Tsit5(),u0=u0,p=params_all,saveat=saveat))
solution = (solution[1:N+1, :] + solution[N+2:end, :])

# Define loss function
function loss_func(p1,p2,p3,ϵ)
    params_all = [p1;p2;p3;ϵ]
    sol_cme = solve(ODEProblem(CME, u0, tspan, params_all),Tsit5(),u0=u0,p=params_all,saveat=saveat)
    temp = sol_cme.u

    solution = (sol_cme[1:N+1, :] + sol_cme[N+2:end, :])
    mse = Flux.mse(Array(solution),train_sol)
    print("mse:",mse," ")

    μ_logσ_list = [split_encoder_result(re1(p1)(temp[i]), latent_size) for i=1:length(saveat)]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:length(saveat)])/length(saveat)
    print("kl:",kl,"\n")

    # solution = (sol_cme[1:N+1, :] + sol_cme[N+2:end, :])
    # reg_zero = Flux.mse(Array(solution),train_sol)
    # print("mse:",reg_zero," ")

    loss = λ*mse + kl
    print(loss,"\n")
    return loss
end

λ2 = 6000

ϵ = zeros(latent_size)
loss_func(params1,params2_1,params2_2,ϵ)
grads = gradient(()->loss_func(params1,params2_1,params2_2,ϵ),ps)

# Training process
epochs_all = 0
lr = 0.01;
opt= ADAM(lr);
epochs = 20;
epochs_all = epochs_all + epochs
print("learning rate = ",lr)
@time for epoch in 1:epochs
    ϵ = rand(Normal(),latent_size)
    print(epoch,"\n")
    grads = gradient(()->loss_func(params1,params2_1,params2_2,ϵ),ps)
    Flux.update!(opt, ps, grads)
end

# write parameters
df = DataFrame(params1 = params1,
params2_1 = vcat(params2_1,[0 for i=1:length(params1)-length(params2_1)]),
params2_2 = vcat(params2_2,[0 for i=1:length(params1)-length(params2_2)]))
CSV.write("On-Off/params_trained.csv",df)

# Check
using CSV,DataFrames
df = CSV.read("On-Off/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2_1 = df.params2_1[1:length(params2_1)]
params2_2 = df.params2_2[1:length(params2_2)]
ps = Flux.params(params1,params2_1,params2_2);

u0 = [1.; zeros(2*N+1)]
tf = 10; #end time
tspan = (0, tf);
saveat = 0:0.1:10
ϵ = zeros(latent_size)
params_all = [params1;params2_1;params2_2;ϵ];
problem = ODEProblem(CME, u0, tspan, params_all);
solution = Array(solve(problem,Tsit5(),u0=u0,p=params_all,saveat=saveat))
solution = (solution[1:N+1, :] + solution[N+2:end, :]);

Flux.mse(solution,train_sol)

# Check mean value
mean_exact = [sum([j for j=0:N].*train_sol[:,i]) for i=1:size(train_sol,2)]
mean_trained = [sum([j for j=0:N].*solution[:,i]) for i=1:size(solution,2)]
plot(mean_trained,linewidth = 3,label="VAE-CME",xlabel = "# t", ylabel = "mean value")
plot!(mean_exact,label="SSA",linewidth = 3,line=:dash,legend=:bottomright)

# Check probability distribution
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label="SSA",title=join(["t=",time_choose/10]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(5)
    p2 = plot_distribution(10)
    p3 = plot_distribution(15)
    p4 = plot_distribution(20)
    p5 = plot_distribution(30)
    p6 = plot_distribution(40)
    p7 = plot_distribution(50)
    p8 = plot_distribution(60)
    p9 = plot_distribution(70)
    p10 = plot_distribution(80)
    p11 = plot_distribution(90)
    p12 = plot_distribution(100)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
# savefig("On-Off/fitting.svg")

#=
function plot_all()
    time_choose = 5
    p1=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label="VAE-CME", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label="Exact",title=join(["t=",time_choose/10]),line=:dash)
    
    time_choose = 10
    p2=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose/10]),line=:dash)
    
    time_choose = 15
    p3=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose/10]),line=:dash)

    time_choose = 20
    p4=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose/10]),line=:dash)

    time_choose = 30
    p5=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false, ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose/10]),line=:dash)

    time_choose = 40
    p6=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose/10]),line=:dash)

    time_choose = 50
    p7=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose/10]),line=:dash)
    
    time_choose = 60
    p8=plot(0:N,solution[:,time_choose+1],xticks=false,linewidth = 3,label=false)
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose/10]),line=:dash)

    time_choose = 70
    p9=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose/10]),line=:dash)

    time_choose = 80
    p10=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose/10]),line=:dash)

    time_choose = 90
    p11=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose/10]),line=:dash)

    time_choose = 100
    p12=plot(0:N,solution[:,time_choose+1],linewidth = 3,label=false,xlabel = "# of products")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label=false,title=join(["t=",time_choose/10]),line=:dash)

    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()
savefig("Statement/Figs/On-Off_fitting.pdf")


