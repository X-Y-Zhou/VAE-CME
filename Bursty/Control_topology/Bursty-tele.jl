using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

N = 64
function bursty(N,a,b,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

a = 0.0282
b = 3.46

end_time = 150
train_sol = zeros(N+1,end_time+1)
for i = 0:end_time
    if i < 120
        train_sol[1:N+1,i+1] = bursty(N+1,a,b,i)
    else
        train_sol[1:N+1,i+1] = bursty(N+1,a,b,120)
    end
end

sigma_on=0.0282;
sigma_off=1.0;
rho_on=3.46; 
rho_off=0.0;
gamma=0.0;

# # Model initialization
# latent_size = 5;
# encoder = Chain(Dense(2*(N+1), 10,tanh),Dense(10, latent_size * 2));
# decoder_1 = Chain(Dense(latent_size, 10,tanh),Dense(10, N),x-> x.+[i/1 for i in 1:N],x ->relu.(x));
# decoder_2  = Chain(Dense(latent_size, 10,tanh),Dense(10, N),x-> x.+[i/1 for i in 1:N],x ->relu.(x));

# params1, re1 = Flux.destructure(encoder);
# params2_1, re2_1 = Flux.destructure(decoder_1);
# params2_2, re2_2 = Flux.destructure(decoder_2);
# ps = Flux.params(params1,params2_1,params2_2);

# model initialization
latent_size = 2;
encoder = Chain(Dense(N, 200,tanh),Dense(200, latent_size * 2));
decoder = Chain(Dense(latent_size, 200),Dense(200 , 4),x ->exp.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

# Define the CME
function CME(du, u, p, t)
    h = re1(p[1:length(params1)])(u[1:N])
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
    l,m,n,o = re2(p[length(params1)+1:end-latent_size])(z)
    NN1 = f_NN.(1:N-1,l,m,n,o)
    # NN1 = re2_1(p[length(params1)+1:length(params1)+length(params2_1)])(z)
    # NN2 = re2_2(p[length(params1)+length(params2_1)+1:end-latent_size])(z)

    h = re1(p[1:length(params1)])(u[N+1:2*N])
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
    l,m,n,o = re2(p[length(params1)+1:end-latent_size])(z)
    NN2 = f_NN.(1:N-1,l,m,n,o)

    du[1] = (-sigma_on-rho_off)*u[1] + (-gamma+NN1[1])*u[2] + sigma_off*u[N+1]
    for i in 2:N-1
        du[i] = rho_off*u[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*u[i] + (-i*gamma+NN1[i])*u[i+1] + sigma_off*u[i+N];
    end
    du[N] = rho_off*u[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*u[N] + sigma_off*u[2*N];

    du[N+1] = sigma_on*u[1] + (-sigma_off-rho_on)*u[N+1] + (-gamma+NN2[1])*u[N+2]
    for i in (N+2):(2*N-1)
        du[i] = sigma_on*u[i-N] + rho_on*u[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*u[i] + (-(i-N)*gamma+NN2[i-N])*u[i+1]
    end
    du[2*N] = sigma_on*u[N] +rho_on*u[2*N-1]+(-sigma_off-rho_on+N*gamma -NN2[N-1])*u[2*N]

    # du[1] = (-sigma_on-rho_off)*u[1] + (-gamma+NN1[1])*u[2] + sigma_off*u[N+2]
    # for i in 2:N
    #     du[i] = rho_off*u[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*u[i] + (-i*gamma+NN1[i])*u[i+1] + sigma_off*u[i+N+1];
    # end
    # du[N+1] = rho_off*u[N] + (-sigma_on-rho_off+N*gamma-NN1[N])*u[N+1] + sigma_off*u[2*N+2];

    # du[N+2] = sigma_on*u[1] + (-sigma_off-rho_on)*u[N+2] + (-gamma+NN2[1])*u[N+3]
    # for i in (N+3):(2*N+1)
    #     du[i] = sigma_on*u[i-N-1] + rho_on*u[i-1] + (-sigma_off-rho_on+(i-N-2)*gamma -NN2[i-N-2])*u[i] + (-(i-N-1)*gamma+NN2[i-N-1])*u[i+1]
    # end
    # du[2*N+2] = sigma_on*u[N+1] +rho_on*u[2*N+1]+(-sigma_off-rho_on+N*gamma -NN2[N])*u[2*N+2]
end

# Initialize the ODE solver
u0 = [1.; zeros(2*N-1)]
tf = 150;
tspan = (0, tf);
saveat = 0:1:150
ϵ = zeros(latent_size)
params_all = [params1;params2;ϵ];
problem = ODEProblem(CME, u0, tspan, params_all);
solution = Array(solve(problem,Tsit5(),u0=u0,p=params_all,saveat=saveat))
solution = (solution[1:N, :] + solution[N+1:end, :])

# Check
using CSV,DataFrames
df = CSV.read("Bursty/Control_rate_Inference/control_kinetic/params_ck7_better.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

u0 = [1.; zeros(2*N-1)]
tf = 150;
tspan = (0, tf);
saveat = 0:1:150
ϵ = zeros(latent_size)
params_all = [params1;params2;ϵ];
problem = ODEProblem(CME, u0, tspan, params_all);
solution = Array(solve(problem,Tsit5(),u0=u0,p=params_all,saveat=saveat))
solution = (solution[1:N, :] + solution[N+1:end, :])

mean_exact = [sum([j for j=0:N].*train_sol[:,i]) for i=1:size(train_sol,2)]
mean_trained = [sum([j for j=0:N-1].*solution[:,i]) for i=1:size(solution,2)]
plot(mean_trained,linewidth = 3,label="VAE-CME",xlabel = "# t", ylabel = "mean value")
plot!(mean_exact,label="exact",linewidth = 3,line=:dash,legend=:bottomright)

# Check probability distribution
function plot_distribution(time_choose)
    p=plot(0:N-1,solution[:,time_choose+1],linewidth = 3,label="VAE-CME-topology",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,train_sol[:,time_choose+1],linewidth = 3,label="SSA",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(10)
    p2 = plot_distribution(20)
    p3 = plot_distribution(30)
    p4 = plot_distribution(40)
    p5 = plot_distribution(50)
    p6 = plot_distribution(60)
    p7 = plot_distribution(70)
    p8 = plot_distribution(80)
    p9 = plot_distribution(90)
    p10 = plot_distribution(100)
    p11 = plot_distribution(110)
    p12 = plot_distribution(120)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()


