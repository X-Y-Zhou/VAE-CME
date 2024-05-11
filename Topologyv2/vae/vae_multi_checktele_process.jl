using Distributed,Pkg
addprocs(3)
# rmprocs(5)
nprocs()
workers()

@everywhere using Flux, DiffEqSensitivity, DifferentialEquations
@everywhere using Distributions, Distances,Random
@everywhere using DelimitedFiles, Plots

@everywhere include("../../utils.jl")

@everywhere τ = 10
@everywhere N = 120

# tele params and check_sol
@everywhere version = 1
@everywhere ps_matrix_tele = readdlm("Topologyv2/tele/data/ps_telev$version.txt")
@everywhere sigma_on_list = ps_matrix_tele[1,:]
@everywhere sigma_off_list = ps_matrix_tele[2,:]
@everywhere rho_on_list = ps_matrix_tele[3,:]
@everywhere rho_off = 0.0
@everywhere gamma= 0.0
@everywhere batchsize_tele = size(ps_matrix_tele,2)
check_sol = readdlm("Topologyv2/tele/data/matrix_telev$version.csv")

# bd params and train_sol
@everywhere ps_matrix_bd = vec(readdlm("Topologyv2/ps_bdv1.csv"))
@everywhere ρ_list = ps_matrix_bd
@everywhere batchsize_bd = length(ρ_list)
train_sol = hcat([birth_death(N, ρ_list[i], τ) for i = 1:length(ρ_list)]...)

# model initialization
@everywhere latent_size = 2;
@everywhere encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
@everywhere decoder = Chain(Dense(latent_size, 10),Dense(10 , N-1),x -> x.+[i/τ  for i in 1:N-1],x ->relu.(x));

@everywhere params1, re1 = Flux.destructure(encoder);
@everywhere params2, re2 = Flux.destructure(decoder);
@everywhere ps = Flux.params(params1,params2);

# 双峰 x -> x.+[i/τ  for i in 1:N-1] 
# bino ρ较大时 注意截断

# CME
@everywhere function f1!(x,p1,p2,ϵ,ρ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN = re2(p2)(z)
    return vcat(-ρ*x[1] + NN[1]*x[2],
                [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
                sum(x)-1)
end

@everywhere sol_bd(p1,p2,ϵ,ρ,P0) = nlsolve(x->f1!(x,p1,p2,ϵ,ρ),P0).zero

@everywhere function solve_bd(ρ,p1,p2,ϵ)
    P_0_distribution = Poisson(ρ*τ)
    P_0 = [pdf(P_0_distribution,i) for i=0:N-1]
    solution = sol_bd(p1,p2,ϵ,ρ,P_0)
    return solution
end

ϵ = zeros(latent_size)
# @time solution_bd = hcat([solve_bd(ρ_list[i],params1,params2,ϵ) for i=1:batchsize]...)
@time solution_bd = hcat(pmap(i->solve_bd(ρ_list[i],params1,params2,ϵ),1:batchsize_bd)...);
mse_bd = Flux.mse(solution_bd,train_sol)

@everywhere function f_tele!(x,p1,p2,ϵ,sigma_on,sigma_off,rho_on)
    
    h = re1(p1)(x[1:N])
    # h = re1(p1)(x[1:N])*sigma_off/(sigma_on+sigma_off)
    # h = re1(p1)(x[1:N]*sigma_off/(sigma_on+sigma_off))

    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN1 = re2(p2)(z)

    # NN1 = NN1*sigma_off/(sigma_on+sigma_off)
    # NN1 = NN1*(sigma_on+sigma_off)/sigma_off


    h = re1(p1)(x[N+1:2*N])
    # h = re1(p1)(x[N+1:2*N])*sigma_on/(sigma_on+sigma_off)
    # h = re1(p1)(x[N+1:2*N]*sigma_on/(sigma_on+sigma_off))

    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN2 = re2(p2)(z)

    # NN2 = NN2*sigma_on/(sigma_on+sigma_off)
    # NN2 = NN2*(sigma_on+sigma_off)/sigma_on

    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)
end

@everywhere sol_tele(p1,p2,ϵ,sigma_on,sigma_off,rho_on,P_0) = nlsolve(x->f_tele!(x,p1,p2,ϵ,sigma_on,sigma_off,rho_on),P_0).zero

# P0,P1
@everywhere function solve_tele(sigma_on,sigma_off,rho_on,p1,p2,ϵ)
    P_0_distribution = Poisson(rho_on*τ*sigma_on)
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
    P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]

    solution = sol_tele(p1,p2,ϵ,sigma_on,sigma_off,rho_on,P_0_split)
    solution = solution[1:N]+solution[N+1:2*N]
    return solution
end

# @time solution_tele = hcat([solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ) for i=1:50]...)
@time solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ),1:batchsize_tele)...);
mse_tele = Flux.mse(solution_tele,check_sol)

@everywhere function loss_func(p1,p2,ϵ)
    sol_cme = hcat(pmap(i->solve_bd(ρ_list[i],p1,p2,ϵ),1:batchsize_bd)...);
    mse = Flux.mse(sol_cme,train_sol)
    print("mse:",mse,"\n")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[:,i]), latent_size) for i=1:batchsize_bd]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:batchsize_bd])/batchsize_bd
    print("kl:",kl,"\n")

    loss = λ*mse + kl
    print("loss:",loss,"\n")
    return loss
end

λ = 1e8
@time loss_bd = loss_func(params1,params2,ϵ)
@time grads = gradient(()->loss_func(params1,params2,ϵ) , ps)
mse_min = [mse_tele]

# training
lr_list = [0.002,0.001,0.0008,0.0006,0.0004,0.0002,0.0001]  #lr需要操作一下的
lr_list = [0.002]
lr_list = [0.01]
mse_min
lr_list = [0.01,0.008,0.006,0.004,0.002,0.001]

for lr in lr_list
    opt= ADAM(lr);
    epochs = 50
    print("learning rate = ",lr,"\n")

    @time for epoch in 1:epochs
        print(epoch,"\n")
        ϵ = rand(Normal(),latent_size)
        grads = gradient(()->loss_func(params1,params2,ϵ) , ps);
        Flux.update!(opt, ps, grads)

        ϵ = zeros(latent_size)
        solution_bd = hcat(pmap(i->solve_bd(ρ_list[i],params1,params2,ϵ),1:batchsize_bd)...);
        mse_bd = Flux.mse(solution_bd,train_sol)

        solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ),1:batchsize_tele)...);
        mse_tele = Flux.mse(solution_tele,check_sol)

        if mse_tele<mse_min[1]
            df = DataFrame(params1 = vcat(params1,[0 for i=1:length(params2)-length(params1)]),params2 = params2)
            CSV.write("Topologyv2/vae/params_trained_vae_tele.csv",df)
            mse_min[1] = mse_tele
        end
        print("mse_bd:",mse_bd,"\n")
        print("mse_tele:",mse_tele,"\n")
    end

    using CSV,DataFrames
    df = CSV.read("Topologyv2/vae/params_trained_vae_tele.csv",DataFrame)
    params1 = df.params1[1:length(params1)]
    params2 = df.params2[1:length(params2)]
    ps = Flux.params(params1,params2);
end

using CSV,DataFrames
df = CSV.read("Topologyv2/vae/params_trained_vae_tele.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);
# end

ϵ = zeros(latent_size)
@time solution_bd = hcat(pmap(i->solve_bd(ρ_list[i],params1,params2,ϵ),1:batchsize_bd)...);
mse_bd = Flux.mse(solution_bd,train_sol)

@time solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ),1:batchsize_tele)...);
mse_tele = Flux.mse(solution_tele,check_sol)
mse_min = [mse_tele]

function plot_distribution(set)
    plot(0:N-1,solution_tele[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,check_sol[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_tele[:,set],digits=4)]),line=:dash)
end
plot_distribution(1)

# function plot_distribution(set)
#     plot(0:N-1,solution_bd[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
#     plot!(0:N-1,train_sol[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_bd[set],digits=3)]),line=:dash)
# end

function plot_channel(i)
    p1 = plot_distribution(1+10*(i-1))
    p2 = plot_distribution(2+10*(i-1))
    p3 = plot_distribution(3+10*(i-1))
    p4 = plot_distribution(4+10*(i-1))
    p5 = plot_distribution(5+10*(i-1))
    p6 = plot_distribution(6+10*(i-1))
    p7 = plot_distribution(7+10*(i-1))
    p8 = plot_distribution(8+10*(i-1))
    p9 = plot_distribution(9+10*(i-1))
    p10 = plot_distribution(10+10*(i-1))
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,layouts=(2,5),size=(1500,600))
end
plot_channel(5)

for i = 1:10
    p = plot_channel(i)
    savefig(p,"Topologyv2/topo_results/fig_$i.svg")
end



