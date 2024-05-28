# 仅用bursty的数据进行外拓
using Distributed,Pkg
addprocs(3)
# rmprocs(5)
nprocs()
workers()

@everywhere using Flux, DiffEqSensitivity, DifferentialEquations
@everywhere using Distributions, Distances,Random
@everywhere using DelimitedFiles, Plots

@everywhere include("../../../utils.jl")

@everywhere τ = 10
@everywhere N = 120

# tele params and check_sol
@everywhere ps_matrix_tele = readdlm("Topology_results/var_delay/tele_data/ps_tele_final.txt")
@everywhere sigma_on_list = ps_matrix_tele[1,:]
@everywhere sigma_off_list = ps_matrix_tele[2,:]
@everywhere rho_on_list = ps_matrix_tele[3,:]
@everywhere rho_off = 0.0
@everywhere gamma= 0.0
@everywhere batchsize_tele = size(ps_matrix_tele,2)
check_sol1 = readdlm("Topology_results/var_delay/tele_data/matrix_tele_final_10-10.txt") # Attrtibute = 0
check_sol2 = readdlm("Topology_results/var_delay/tele_data/matrix_tele_final_0-20.txt") # Attrtibute = 1
check_sol3 = readdlm("Topology_results/var_delay/tele_data/matrix_tele_final_5-15.txt") # Attrtibute = 0.5

# check_sol1 = readdlm("Topologyv6/tele/data/matrix_tele_10-10.csv") # Attrtibute = 0
# check_sol2 = readdlm("Topologyv6/tele/data/matrix_tele_0-20.csv") # Attrtibute = 1
# check_sol3 = readdlm("Topologyv6/tele/data/matrix_tele_5-15.csv") # Attrtibute = 0.5

# bursty params and train_sol
@everywhere ps_matrix_bursty = readdlm("Topology_results/var_delay/bursty_data/ps_burstyv1.txt")
@everywhere batchsize_bursty = size(ps_matrix_bursty,2)
@everywhere a_list = ps_matrix_bursty[1,:]
@everywhere b_list = ps_matrix_bursty[2,:]

train_sol1 = readdlm("Topology_results/var_delay/bursty_data/matrix_bursty_10-10.txt") # var = 0
train_sol2 = readdlm("Topology_results/var_delay/bursty_data/matrix_bursty_0-20.txt") # var = max

# model initialization
@everywhere latent_size = 2;
@everywhere encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
@everywhere decoder_1 = Chain(Dense(latent_size+1, 10),Dense(10 , N-1),x -> x.+[i/τ  for i in 1:N-1],x ->relu.(x));
@everywhere decoder_2 = Chain(decoder_1[1],decoder_1[2],decoder_1[3],decoder_1[4]);

@everywhere params1, re1 = Flux.destructure(encoder);
@everywhere params2, re2_1 = Flux.destructure(decoder_1);
@everywhere       _, re2_2 = Flux.destructure(decoder_2);
@everywhere ps = Flux.params(params1,params2);

# CME
@everywhere function f1!(x,p1,p2,ϵ,a,b)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,0)
    NN = re2_1(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

@everywhere function f2!(x,p1,p2,ϵ,a,b)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,1)
    NN = re2_2(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

@everywhere sol_bursty1(p1,p2,ϵ,a,b,P0) = nlsolve(x->f1!(x,p1,p2,ϵ,a,b),P0).zero
@everywhere sol_bursty2(p1,p2,ϵ,a,b,P0) = nlsolve(x->f2!(x,p1,p2,ϵ,a,b),P0).zero

@everywhere function solve_bursty1(a,b,p1,p2,ϵ)
    P_0_distribution = NegativeBinomial(a*τ, 1/(1+b))
    P_0 = [pdf(P_0_distribution,i) for i=0:N-1]
    solution = sol_bursty1(p1,p2,ϵ,a,b,P_0)
    return solution
end

@everywhere function solve_bursty2(a,b,p1,p2,ϵ)
    P_0_distribution = NegativeBinomial(a*τ, 1/(1+b))
    P_0 = [pdf(P_0_distribution,i) for i=0:N-1]
    solution = sol_bursty2(p1,p2,ϵ,a,b,P_0)
    return solution
end

ϵ = zeros(latent_size)
@time solution_bursty1 = hcat(pmap(i->solve_bursty1(a_list[i],b_list[i],params1,params2,ϵ),1:batchsize_bursty)...);
@time solution_bursty2 = hcat(pmap(i->solve_bursty2(a_list[i],b_list[i],params1,params2,ϵ),1:batchsize_bursty)...);
mse_bursty1 = Flux.mse(solution_bursty1,train_sol1)
mse_bursty2 = Flux.mse(solution_bursty2,train_sol2)

@everywhere function f_tele!(x,p1,p2,ϵ,sigma_on,sigma_off,rho_on,Attribute)
    h = re1(p1)(x[1:N])
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,Attribute)
    NN1 = re2_1(p2)(z)

    h = re1(p1)(x[N+1:2*N])
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,Attribute)
    NN2 = re2_1(p2)(z)
    
    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)
end

@everywhere sol_tele(p1,p2,ϵ,sigma_on,sigma_off,rho_on,P_0,Attrtibute) = nlsolve(x->f_tele!(x,p1,p2,ϵ,sigma_on,sigma_off,rho_on,Attrtibute),P_0).zero

# P0,P1
@everywhere function solve_tele(sigma_on,sigma_off,rho_on,p1,p2,ϵ,Attrtibute)
    P_0_distribution = Poisson(rho_on*τ*sigma_on)
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
    P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]

    solution = sol_tele(p1,p2,ϵ,sigma_on,sigma_off,rho_on,P_0_split,Attrtibute)
    solution = solution[1:N]+solution[N+1:2*N]
    return solution
end

Attrtibute = 0
@time solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
mse_tele = Flux.mse(solution_tele,check_sol1)

@everywhere function loss_func1(p1,p2,ϵ)
    sol_cme = hcat(pmap(i->solve_bursty1(a_list[i],b_list[i],p1,p2,ϵ),1:batchsize_bursty)...);
    mse = Flux.mse(sol_cme,train_sol1)
    print("mse:",mse,"\n")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[:,i]), latent_size) for i=1:batchsize_bursty]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:batchsize_bursty])/batchsize_bursty
    print("kl:",kl,"\n")

    loss = λ1*mse + kl
    print("loss:",loss,"\n")
    return loss
end

@everywhere function loss_func2(p1,p2,ϵ)
    sol_cme = hcat(pmap(i->solve_bursty2(a_list[i],b_list[i],p1,p2,ϵ),1:batchsize_bursty)...);
    mse = Flux.mse(sol_cme,train_sol2)
    print("mse:",mse,"\n")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[:,i]), latent_size) for i=1:batchsize_bursty]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:batchsize_bursty])/batchsize_bursty
    print("kl:",kl,"\n")

    loss = λ2*mse + kl
    print("loss:",loss,"\n")
    return loss
end

function loss_func(p1,p2,ϵ)
    loss = loss_func1(p1,p2,ϵ) + loss_func2(p1,p2,ϵ)
    return loss
end

λ1 = 1e8
λ2 = 1e8
@time loss_bursty = loss_func(params1,params2,ϵ)
@time grads = gradient(()->loss_func(params1,params2,ϵ) , ps)
mse_bursty = mse_bursty1 + mse_bursty2
mse_min = [mse_bursty]

# training
mse_min
lr_list = [0.01,0.008,0.006,0.004,0.002,0.001]
lr_list = [0.006,0.004,0.002,0.001]
lr_list = [0.01,0.008,0.006]
lr_list = [0.01]
lr_list = [0.005]
lr_list = [0.001,0.0008,0.0006,0.0004,0.0002,0.0001]


for lr in lr_list
    opt= ADAM(lr);
    epochs = 60
    print("learning rate = ",lr,"\n")

    @time for epoch in 1:epochs
        print(epoch,"\n")
        ϵ = rand(Normal(),latent_size)
        grads = gradient(()->loss_func(params1,params2,ϵ) , ps);
        Flux.update!(opt, ps, grads)

        ϵ = zeros(latent_size)
        # solution_bursty1 = hcat(pmap(i->solve_bursty1(a_list[i],b_list[i],params1,params2,ϵ),1:batchsize_bursty)...);
        # solution_bursty2 = hcat(pmap(i->solve_bursty2(a_list[i],b_list[i],params1,params2,ϵ),1:batchsize_bursty)...);

        # mse_bursty1 = Flux.mse(solution_bursty1,train_sol1)
        # mse_bursty2 = Flux.mse(solution_bursty2,train_sol2)
        # mse_bursty = mse_bursty1 + mse_bursty2

        # Attrtibute = 0
        # solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
        # mse_tele = Flux.mse(solution_tele,check_sol)
        
        # Attrtibute = 0
        # solution_tele1 = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
        # mse_tele1 = Flux.mse(solution_tele1,check_sol1)

        Attrtibute = 1
        solution_tele2 = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
        mse_tele2 = Flux.mse(solution_tele2,check_sol2)
        mse_tele = mse_tele2

        if mse_tele<mse_min[1]
            df = DataFrame(params1 = vcat(params1,[0 for i=1:length(params2)-length(params1)]),params2 = params2)
            CSV.write("Topology_results/var_delay/vae/params_trained_vae_tele_burstyv3.csv",df)
            mse_min[1] = mse_tele
        end
        # print("mse_bursty1:",mse_bursty1,"\n")
        # print("mse_bursty2:",mse_bursty2,"\n")
        print("mse_tele:",mse_tele,"\n")
    end

    using CSV,DataFrames
    df = CSV.read("Topology_results/var_delay/vae/params_trained_vae_tele_burstyv3.csv",DataFrame)
    params1 = df.params1[1:length(params1)]
    params2 = df.params2[1:length(params2)]
    ps = Flux.params(params1,params2);
end

using CSV,DataFrames
df = CSV.read("Topology_results/var_delay/vae/params_trained_vae_tele_burstyv3.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);
# end

mse_min

ϵ = zeros(latent_size)
@time solution_bursty1 = hcat(pmap(i->solve_bursty1(a_list[i],b_list[i],params1,params2,ϵ),1:batchsize_bursty)...);
@time solution_bursty2 = hcat(pmap(i->solve_bursty2(a_list[i],b_list[i],params1,params2,ϵ),1:batchsize_bursty)...);
mse_bursty1 = Flux.mse(solution_bursty1,train_sol1)
mse_bursty2 = Flux.mse(solution_bursty2,train_sol2)
mse_bursty = mse_bursty1 + mse_bursty2
# mse_min = [mse_bursty]

# writedlm("Topology_results/var_delay/train_results/pre_Attr=0.txt",solution_bursty1)
# writedlm("Topology_results/var_delay/train_results/pre_Attr=1.txt",solution_bursty2)

Attrtibute = 0
@time solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
solution_tele1 = solution_tele
check_sol = check_sol1
mse_tele1 = Flux.mse(solution_tele,check_sol)

Attrtibute = 1
@time solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
solution_tele2 = solution_tele
check_sol = check_sol2
mse_tele2 = Flux.mse(solution_tele,check_sol)
# mse_min = [mse_tele2]
# mse_min = [mse_tele1+mse_tele2]
# Flux.mse(solution_tele[:,21:50],check_sol[:,21:50])

Attrtibute = 0.5
@time solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
solution_tele3 = solution_tele
check_sol = check_sol3
mse_tele2 = Flux.mse(solution_tele,check_sol)

writedlm("Topology_results/var_delay/topo_results/pre_Attr=0.txt",solution_tele1)
writedlm("Topology_results/var_delay/topo_results/pre_Attr=1.txt",solution_tele2)
writedlm("Topology_results/var_delay/topo_results/pre_Attr=0.5.txt",solution_tele3)


# function plot_distribution(set)
#     plot(0:N-1,solution_tele3[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
#     plot!(0:N-1,check_sol3[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_tele[:,set],digits=4)]),line=:dash)
# end

function plot_distribution(set)
    plot(0:N-1,solution_bursty2[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol2[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_bursty[:,set],digits=4)]),line=:dash)
end

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
plot_channel(1)

for i = 1:5
    p = plot_channel(i)
    savefig(p,"Topology_results/var_delay/train_results/fig_Attri=1_$i.svg")
end

# select
# bursty
bursty_list = [2,3,5,17,38,48]
ps_matrix_bursty = readdlm("Topology_results/var_delay/bursty_data/ps_burstyv1.txt")[:,bursty_list]

solution_bursty1 = readdlm("Topology_results/var_delay/train_results/pre_Attr=0.txt")[:,bursty_list]
check_bursty1 = readdlm("Topology_results/var_delay/bursty_data/matrix_bursty_10-10.txt")[:,bursty_list]
Flux.mse(solution_bursty1,check_bursty1)

solution_bursty2 = readdlm("Topology_results/var_delay/train_results/pre_Attr=1.txt")[:,bursty_list]
check_bursty2 = readdlm("Topology_results/var_delay/bursty_data/matrix_bursty_0-20.txt")[:,bursty_list]
Flux.mse(solution_bursty2,check_bursty2)

writedlm("Topology_results/var_delay/plot_omni_data/bursty/ps_bursty.txt",ps_matrix_bursty)
writedlm("Topology_results/var_delay/plot_omni_data/bursty/pre_Attr=0.txt",solution_bursty1)
writedlm("Topology_results/var_delay/plot_omni_data/bursty/pre_Attr=1.txt",solution_bursty2)

writedlm("Topology_results/var_delay/plot_omni_data/bursty/SSA_Attr=0.txt",check_bursty1)
writedlm("Topology_results/var_delay/plot_omni_data/bursty/SSA_Attr=1.txt",check_bursty2)

# tele
tele_list = [6,17,21,28,35,50]
ps_matrix_tele = readdlm("Topology_results/var_delay/tele_data/ps_tele_final.txt")[:,tele_list]

solution_tele1 = readdlm("Topology_results/var_delay/topo_results/pre_Attr=0.txt")[:,tele_list]
check_tele1 = readdlm("Topology_results/var_delay/tele_data/matrix_tele_final_10-10.txt")[:,tele_list]
Flux.mse(solution_tele1,check_tele1)

solution_tele2 = readdlm("Topology_results/var_delay/topo_results/pre_Attr=1.txt")[:,tele_list]
check_tele2 = readdlm("Topology_results/var_delay/tele_data/matrix_tele_final_0-20.txt")[:,tele_list]
Flux.mse(solution_tele2,check_tele2)

writedlm("Topology_results/var_delay/plot_omni_data/tele/ps_tele.txt",ps_matrix_tele)
writedlm("Topology_results/var_delay/plot_omni_data/tele/pre_Attr=0.txt",solution_tele1)
writedlm("Topology_results/var_delay/plot_omni_data/tele/pre_Attr=1.txt",solution_tele2)

writedlm("Topology_results/var_delay/plot_omni_data/tele/SSA_Attr=0.txt",check_tele1)
writedlm("Topology_results/var_delay/plot_omni_data/tele/SSA_Attr=1.txt",check_tele2)

# read selected and check
# bursty
ps_matrix_bursty = readdlm("Topology_results/var_delay/plot_omni_data/bursty/ps_bursty.txt")

solution_bursty1 = readdlm("Topology_results/var_delay/plot_omni_data/bursty/pre_Attr=0.txt")
check_bursty1 = readdlm("Topology_results/var_delay/plot_omni_data/bursty/SSA_Attr=0.txt")
Flux.mse(solution_bursty1,check_bursty1)

solution_bursty2 = readdlm("Topology_results/var_delay/plot_omni_data/bursty/pre_Attr=1.txt")
check_bursty2 = readdlm("Topology_results/var_delay/plot_omni_data/bursty/SSA_Attr=1.txt")
Flux.mse(solution_bursty2,check_bursty2)

# tele
ps_matrix_tele = readdlm("Topology_results/var_delay/plot_omni_data/tele/ps_tele.txt")

solution_tele1 = readdlm("Topology_results/var_delay/plot_omni_data/tele/pre_Attr=0.txt")
check_tele1 = readdlm("Topology_results/var_delay/plot_omni_data/tele/SSA_Attr=0.txt")
Flux.mse(solution_tele1,check_tele1)

solution_tele2 = readdlm("Topology_results/var_delay/plot_omni_data/tele/pre_Attr=1.txt")
check_tele2 = readdlm("Topology_results/var_delay/plot_omni_data/tele/SSA_Attr=1.txt")
Flux.mse(solution_tele2,check_tele2)

# plot
function plot_distribution(set)
    plot(0:N-1,solution_bursty2[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,check_bursty2[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_bursty[:,set],digits=4)]),line=:dash)
end

function plot_distribution(set)
    plot(0:N-1,solution_tele2[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,check_tele2[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_tele[:,set],digits=4)]),line=:dash)
end

function plot_channel()
    p1 = plot_distribution(1)
    p2 = plot_distribution(2)
    p3 = plot_distribution(3)
    p4 = plot_distribution(4)
    p5 = plot_distribution(5)
    p6 = plot_distribution(6)
    plot(p1,p2,p3,p4,p5,p6,layouts=(3,2),size=(600,900))
end
plot_channel()

