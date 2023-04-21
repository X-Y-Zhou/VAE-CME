using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../utils.jl")

solnet_X = readdlm("Oscillation/data/X_oringinal.csv",',')[2:end,:]
solnet_Y = readdlm("Oscillation/data/Y_oringinal.csv",',')[2:end,:]
N = 26

train_sol_X = zeros(N+1,size(solnet_X,1))
for i =1:size(solnet_X,1)
    probability = convert_histo(vec(solnet_X[i,:]))[2]
    if length(probability)<N+1
        train_sol_X[1:length(probability),i] = probability
    else
        train_sol_X[1:N+1,i] = probability[1:N+1]
    end
end

train_sol_Y = zeros(N+1,size(solnet_Y,1))
for i =1:size(solnet_Y,1)
    probability = convert_histo(vec(solnet_Y[i,:]))[2]
    if length(probability)<N+1
        train_sol_Y[1:length(probability),i] = probability
    else
        train_sol_Y[1:N+1,i] = probability[1:N+1]
    end
end

k_1S=1.
k_d=1.
k_p=2.
k_2E_T=1.
Km=1.

J1(Y) = k_1S * k_d^k_p / (k_d^k_p + Y^k_p)
J2(Y) = k_2E_T / (Km + Y)
D1(m) = diagm(-1=>fill(J1(m), N)) .+ diagm(0=>[fill(-J1(m), N);0.0])
D2(m) = m*J2(m) * diagm(0=>fill(1.0, N+1))

# model initialization
bias = zeros(N*(N-1))
for i in 0:N-1
        bias[1+i*(N-1):N-1+i*(N-1)] = [i/10 for i in 1:N-1]
end

#model initialization
latent_size = 10;
encoder = Chain(Dense((N+1)*(N+1), 5,tanh),Dense(5, latent_size * 2));
decoder = Chain(Dense(latent_size, 5,tanh),Dense(5, N*(N-1)), x -> x .+ bias, x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

function CME(du, u, p, t)
    h = re1(p[1:length(params1)])(u)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, p[end-latent_size+1:end])
    NN = re2(p[length(params1)+1:end-latent_size])(z)

    Na(k) = diagm(0=>[0.0; NN[(N-1)*k+1:(N-1)*k+N-1]; 0.0])
    Nb(k) = diagm(1=>[NN[(N-1)*k+1:(N-1)*k+N-1]; 0.0])
    # m = 1
    du[1:N+1] = (D1(0)-D2(0)-Na(0)) * u[1:(N+1)] + D2(1) * u[(N+1+1):(N+1+N+1)]
    for m in 2:N
        du[((N+1)*(m-1)+1):((N+1)*(m-1)+N+1)] = Nb(m-2) * u[((N+1)*(m-2)+1):((N+1)*(m-2)+N+1)] +
                                                                                (D1(m-1)-D2(m-1)-Na(m-1)) * u[((N+1)*(m-1)+1):((N+1)*(m-1)+N+1)] +
                                                                                D2(m) * u[((N+1)*m+1):((N+1)*m+N+1)]
    end
    # m =N+1
    du[((N+1)*N+1):((N+1)*N+N+1)] = (Nb(N-1)) * u[(N+1)*(N-1)+1:(N+1)*(N-1)+N+1] + (D1(N)-D2(N))*u[(N+1)*N+1:(N+1)*N+N+1]
end

u0 = [1.; zeros((N+1)*(N+1)-1)]; 
tf = 200.0;
tspan = (0.0, tf);

saveat = 1:2:200
# saveat = vcat(1:1:24,25:5:200)

ϵ = zeros(latent_size)
params_all = [params1;params2;ϵ]
problem = ODEProblem(CME, u0, tspan, params_all);
solution = solve(problem, Tsit5(), u0=u0, p=params_all, saveat=saveat)

function loss_func(p1,p2,ϵ)
    params_all = [p1;p2;ϵ]
    sol_cme = solve(ODEProblem(CME, u0, tspan, params_all),
                Tsit5(),u0=u0,p=params_all,saveat=saveat)

    temp = sol_cme.u
    temp = set_one.(temp)

    μ_logσ_list = [split_encoder_result(re1(p1)(temp[i]), latent_size) for i=1:length(saveat)]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:length(saveat)])/length(saveat)
    print("kl:",kl,"\n")

    solution_time_points = Array(sol_cme)
    
    mse_X = sum([Flux.mse(vec(sum(reshape(solution_time_points[:,i],N+1,N+1),dims=2)),train_sol_X[:,saveat[i]+1]) 
                    for i=1:length(saveat)])/length(saveat)
    mse_Y = sum([Flux.mse(vec(sum(reshape(solution_time_points[:,i],N+1,N+1),dims=1)),train_sol_Y[:,saveat[i]+1]) 
                    for i=1:length(saveat)])/length(saveat)
    smooth_Y = sum([sum(curvature_change_rate(vec(sum(reshape(solution_time_points[:,i],N+1,N+1),dims=1))))
                for i=1:length(saveat)])/length(saveat)
    print("mse:",mse_X," ",mse_Y," ",mse_X+mse_Y,"\n","smooth_Y:",smooth_Y,"\n")
    
    #loss = kl + λ1*(mse_X+mse_Y)
    loss = kl + λ1*mse_Y + λ2*smooth_Y

    print("loss:",loss,"\n")
    return loss
end

λ1 = 100000
λ2 = 10^-3
loss_func(params1,params2,ϵ)

ϵ = rand(Normal(),latent_size)
grads = gradient(()->loss_func(params1,params2,ϵ) , ps)

epochs_all = 0

lr = 0.004;
opt= ADAM(lr);
epochs = 10;
epochs_all = epochs_all + epochs
print("learning rate = ",lr)
@time for epoch in 1:epochs
    ϵ = rand(Normal(),latent_size)
    print(epoch,"\n")
    grads = gradient(()->loss_func(params1,params2,ϵ) , ps)
    Flux.update!(opt, ps, grads)

    solution_time_points = Array(solve(ODEProblem(CME, u0, tspan, [params1;params2;zeros(latent_size)]), 
                            Tsit5(), p=[params1;params2;zeros(latent_size)],u0=u0, saveat=0:1:Int(tf)))
    mse_X = sum([Flux.mse(vec(sum(reshape(solution_time_points[:,i+1],N+1,N+1),dims=2)),train_sol_X[:,i+1]) 
                            for i in saveat])/length(saveat)
    mse_Y = sum([Flux.mse(vec(sum(reshape(solution_time_points[:,i+1],N+1,N+1),dims=1)),train_sol_Y[:,i+1]) 
                            for i in saveat])/length(saveat)
    print("mse ϵ=0:",mse_X," ",mse_Y," ",mse_X+mse_Y,"\n")
end

u0 = [1.; zeros((N+1)*(N+1)-1)]; 
tf = 200.0;
tspan = (0.0, tf);
time_step = 1.0
ϵ = zeros(latent_size)
params_all = [params1;params2;ϵ];
problem = ODEProblem(CME, u0, tspan, params_all);
solution = Array(solve(problem, Tsit5(), u0=u0, p=params_all, saveat=0:time_step:Int(tf)))

#SSA
mean_train_X = [sum([j for j=0:N].*train_sol_X[:,i]) for i=1:size(train_sol_X,2)]
mean_train_Y = [sum([j for j=0:N].*train_sol_Y[:,i]) for i=1:size(train_sol_Y,2)]

tmax = Int(tf+1)
sol_X = zeros(N+1,tmax)
sol_Y = zeros(N+1,tmax)
for i = 1:tmax
    sol_X[:,i]=sum(reshape(solution[:,i],N+1,N+1),dims=2)[1:N+1]
    sol_Y[:,i]=sum(reshape(solution[:,i],N+1,N+1),dims=1)[1:N+1]
end
sol_X
sol_Y

mean_trained_X = [sum([j for j=0:N].*sol_X[:,i]) for i=1:size(sol_X,2)]
mean_trained_Y = [sum([j for j=0:N].*sol_Y[:,i]) for i=1:size(sol_Y,2)]
plot_mean()
plot_distribution_X_all()
plot_distribution_Y_all()

#read params
using CSV,DataFrames
df = CSV.read("Oscillation/smooth/params_origin_trained_y.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

#write params
using DataFrames,CSV
params1
params2
df = DataFrame( params1 = vcat(params1,[0 for i=1:length(params2)-length(params1)]),params2 = params2)
CSV.write("Oscillation/smooth/params_origin_trained_y.csv",df)

function plot_distribution_X(time_choose)
    p=plot(0:N,sol_X[:,time_choose+1],label="X",linewidth = 3,xlabel = "# of products", ylabel = "Probability")
    plot!(0:N,train_sol_X[:,time_choose+1],linewidth = 3,label="SSA",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_distribution_Y(time_choose)
    p=plot(0:N,sol_Y[:,time_choose+1],label="Y",linewidth = 3,xlabel = "# of products", ylabel = "Probability")
    plot!(0:N,train_sol_Y[:,time_choose+1],linewidth = 3,label="SSA",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_mean()
    p1=plot(mean_trained_X,label="X",linewidth = 3,xlabel = "# t", ylabel = "mean value")
    plot!(mean_train_X,label="SSA",linewidth = 3,line=:dash,legend=:bottomright)
    p2=plot(mean_trained_Y,label="Y",linewidth = 3,xlabel = "# t", ylabel = "mean value")
    plot!(mean_train_Y,label="SSA",linewidth = 3,line=:dash,legend=:bottomright)
    plot(p1,p2,size=(1200,400))
end
plot_mean()

function plot_distribution_X_all()
    p1 = plot_distribution_X(5)
    p2 = plot_distribution_X(10)
    p3 = plot_distribution_X(11)
    p4 = plot_distribution_X(13)
    p5 = plot_distribution_X(15)
    p6 = plot_distribution_X(17)
    p7 = plot_distribution_X(19)
    p8 = plot_distribution_X(21)
    p9 = plot_distribution_X(23)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,size=(1200,800))
end
plot_distribution_X_all()

function plot_distribution_X_all()
    p1 = plot_distribution_X(25)
    p2 = plot_distribution_X(27)
    p3 = plot_distribution_X(30)
    p4 = plot_distribution_X(40)
    p5 = plot_distribution_X(50)
    p6 = plot_distribution_X(75)
    p7 = plot_distribution_X(100)
    p8 = plot_distribution_X(150)
    p9 = plot_distribution_X(200)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,size=(1200,800))
end
plot_distribution_X_all()
savefig("Oscillation/smooth/y_X.pdf")

function plot_distribution_Y_all()
    p1 = plot_distribution_Y(5)
    p2 = plot_distribution_Y(10)
    p3 = plot_distribution_Y(11)
    p4 = plot_distribution_Y(13)
    p5 = plot_distribution_Y(15)
    p6 = plot_distribution_Y(17)
    p7 = plot_distribution_Y(19)
    p8 = plot_distribution_Y(21)
    p9 = plot_distribution_Y(23)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,size=(1200,800))
end
plot_distribution_Y_all()

function plot_distribution_Y_all()
    p1 = plot_distribution_Y(25)
    p2 = plot_distribution_Y(27)
    p3 = plot_distribution_Y(30)
    p4 = plot_distribution_Y(40)
    p5 = plot_distribution_Y(50)
    p6 = plot_distribution_Y(75)
    p7 = plot_distribution_Y(100)
    p8 = plot_distribution_Y(150)
    p9 = plot_distribution_Y(200)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,size=(1200,800))
end
plot_distribution_Y_all()

function Derivative_approxi(vec)
    return [vec[i+1]-vec[i] for i=1:length(vec)-1]
end

function curvature(vec1,vec2)
    return [abs(vec2[i])/(1+vec1[i])^(3/2) for i=1:length(vec2)]
end

function change_rate(vec)
    return [abs(vec[i+1]-vec[i]) for i=1:length(vec)-1]
end

function curvature_change_rate(vec)
    first_grads = Derivative_approxi(vec)
    second_grads = Derivative_approxi(first_grads)
    curves = curvature(first_grads,second_grads)
    change_rates = change_rate(curves)
    return change_rates
end

sol_X[:,end]
train_sol_X[:,end]

change_rate_trained = sum(curvature_change_rate(sol_X[:,100]))
change_rate_SSA = sum(curvature_change_rate(train_sol_X[:,end]))
Flux.mse(change_rate_trained,change_rate_SSA)