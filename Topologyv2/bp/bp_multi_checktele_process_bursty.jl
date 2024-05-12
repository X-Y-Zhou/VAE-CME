# 仅用bursty的数据进行外拓
using Distributed,Pkg
addprocs(1)
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

# bursty params and train_sol
@everywhere ps_matrix_bursty = readdlm("Topologyv2/ps_burstyv1.csv")
@everywhere batchsize_bursty = size(ps_matrix_bursty,2)
@everywhere a_list = ps_matrix_bursty[1,:]
@everywhere b_list = ps_matrix_bursty[2,:]
train_sol = hcat([bursty(N, a_list[i],b_list[i], τ) for i = 1:batchsize_bursty]...)

# model initialization
@everywhere model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> x .+ [i/τ for i in 1:N-1], x -> relu.(x));
@everywhere p1, re = Flux.destructure(model);
@everywhere ps = Flux.params(p1);

#CME
@everywhere function f1!(x,p,a,b)
    NN = re(p)(x)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

@everywhere sol_bursty(p,a,b,P0) = nlsolve(x->f1!(x,p,a,b),P0).zero

@everywhere function solve_bursty(a,b,p)
    P_0_distribution = NegativeBinomial(a*τ, 1/(1+b))
    P_0 = [pdf(P_0_distribution,i) for i=0:N-1]
    solution = sol_bursty(p,a,b,P_0)
    return solution
end

# @time solution_bursty = hcat([solve_bursty(a,b_list[i],p1) for i=1:batchsize]...)
@time solution_bursty = hcat(pmap(i->solve_bursty(a_list[i],b_list[i],p1),1:batchsize_bursty)...);
mse_bursty = Flux.mse(solution_bursty,train_sol)

@everywhere function f_tele!(x,p,sigma_on,sigma_off,rho_on)
    NN1 = re(p)(x[1:N])
    NN2 = re(p)(x[N+1:2*N])

    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)
end

@everywhere sol_tele(p,sigma_on,sigma_off,rho_on,P_0) = nlsolve(x->f_tele!(x,p,sigma_on,sigma_off,rho_on),P_0).zero

# P0,P1
@everywhere function solve_tele(sigma_on,sigma_off,rho_on,p)
    P_0_distribution = Poisson(rho_on*τ*sigma_on)
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
    P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]

    solution = sol_tele(p,sigma_on,sigma_off,rho_on,P_0_split)
    solution = solution[1:N]+solution[N+1:2*N]
    return solution
end

# @time solution_tele = hcat([solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],p1) for i=1:batchsize]...)
@time solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],p1),1:batchsize_tele)...);
mse_tele = Flux.mse(solution_tele,check_sol)

function loss_func(p)
    sol_cme = hcat(pmap(i->solve_bursty(a_list[i],b_list[i],p),1:batchsize_bursty)...);
    # sol_cme = hcat([sol(p,a_list[i],b_list[i],P_0_list[i]) for i=1:batchsize]...)
    mse = Flux.mse(sol_cme,train_sol)
    loss = mse
    return loss
end

@time mse_bursty = loss_func(p1)
@time grads = gradient(()->loss_func(p1) , ps)
mse_min = [mse_tele]

# training
lr_list = [0.01]  #lr需要操作一下的

lr_list = [0.01,0.008,0.006,0.004,0.002,0.001]

for lr in lr_list
    opt= ADAM(lr);
    epochs = 50
    print("learning rate = ",lr,"\n")

    @time for epoch in 1:epochs
        print(epoch,"\n")
        grads = gradient(()->loss_func(p1) , ps);
        Flux.update!(opt, ps, grads)

        mse_bursty = loss_func(p1);

        solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],p1),1:batchsize_tele)...);
        mse_tele = Flux.mse(solution_tele,check_sol)

        if mse_tele<mse_min[1]
            df = DataFrame(p1 = p1)
            CSV.write("Topologyv2/bp/params_trained_bp_tele_bursty.csv",df)
            mse_min[1] = mse_tele
        end
        print("mse_bursty:",mse_bursty,"\n")
        print("mse_tele:",mse_tele,"\n")
    end

    using CSV,DataFrames
    df = CSV.read("Topologyv2/bp/params_trained_bp_tele_bursty.csv",DataFrame)
    p1 = df.p1
    ps = Flux.params(p1);
end

using CSV,DataFrames
df = CSV.read("Topologyv2/bp/params_trained_bp_tele_bursty.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);
# end

@time solution_bursty = hcat(pmap(i->solve_bursty(a_list[i],b_list[i],p1),1:batchsize_bursty)...);
mse_bursty = Flux.mse(solution_bursty,train_sol)

@time solution_tele = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],p1),1:batchsize_tele)...);
mse_tele = Flux.mse(solution_tele,check_sol)
mse_min = [mse_tele]

function plot_distribution(set)
    plot(0:N-1,solution_tele[:,set],linewidth = 3,label="NN-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,check_sol[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_tele[:,set],digits=4)]),line=:dash)
end
plot_distribution(1)

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

for i = 1:10
    p = plot_channel(i)
    savefig(p,"Topologyv2/topo_results/fig_$i.svg")
end