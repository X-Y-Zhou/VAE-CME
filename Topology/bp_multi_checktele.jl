using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances,Random
using DelimitedFiles, Plots

include("../utils.jl")

# tele params
seed = 1
rng = Random.seed!(seed)
sigma_on_list =  [rand(rng,Uniform(0.002,0.005),30);rand(rng,Uniform(0.002,0.005),20)]
sigma_off_list = [rand(rng,Uniform(0.002,0.005),30);rand(rng,Uniform(0.002,0.005),20)]
rho_on_list =    [rand(rng,Uniform(0.1,1),50);]
batchsize = length(rho_on_list)
ps_list = [[sigma_on_list[i],sigma_off_list[i],rho_on_list[i]] for i=1:batchsize]

# bd params
seed = 1
rng = Random.seed!(seed)
ρ_list = rand(rng,Uniform(0.1,1),50)
batchsize = length(ρ_list)

τ = 40
N = 80

train_sol = hcat([birth_death(N, ρ_list[i], τ) for i = 1:length(ρ_list)]...)
check_sol = readdlm("Topology/tele/data/matrix_tele.csv")

# model initialization
model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> x .+ [i/τ for i in 1:N-1], x -> relu.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);

#CME
function f1!(x,p,ρ)
    NN = re(p)(x)
    return vcat(-ρ*x[1] + NN[1]*x[2],
                [ρ*x[i-1] + (-ρ-NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],
                sum(x)-1)
end

#solve P
P_0_distribution_list = [Poisson(ρ_list[i]*τ) for i=1:batchsize]
P_0_list = [[pdf(P_0_distribution_list[j],i) for i=0:N-1] for j=1:batchsize]
sol(p,ρ,P0) = nlsolve(x->f1!(x,p,ρ),P0).zero

@time mse = hcat([sol(p1,ρ_list[i],P_0_list[i]) for i=1:batchsize]...);
@time mse = hcat(sol.(Ref(p1),ρ_list,P_0_list)...);

function f_tele!(x,p,sigma_on,sigma_off,rho_on)
    NN1 = re(p)(x[1:N])
    NN2 = re(p)(x[N+1:2*N])

    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)
end

# P0,P1
function solve_tele(sigma_on,sigma_off,rho_on)
    P_0_distribution = Poisson(rho_on*τ*sigma_on)
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
    P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]

    sol_tele(p,P_0) = nlsolve(x->f_tele!(x,p,sigma_on,sigma_off,rho_on),P_0).zero
    solution = sol_tele(p1,P_0_split)
    solution = solution[1:N]+solution[N+1:2*N]
    return solution
end


function loss_func(p)
    sol_cme = hcat([sol(p,ρ_list[i],P_0_list[i]) for i=1:batchsize]...)
    # sol_cme = hcat(sol.(Ref(p),ρ_list,P_0_list)...)
    mse = Flux.mse(sol_cme,train_sol)
    loss = mse
    return loss
end

using CSV,DataFrames
df = CSV.read("Topology/params_trained_bp.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

@time mse_bd = loss_func(p1)
@time grads = gradient(()->loss_func(p1) , ps)

@time solution_tele = hcat([solve_tele(ps_list[i][1],ps_list[i][2],ps_list[i][3]) for i=1:batchsize]...)
@time solution_tele = hcat([solve_tele.(sigma_on_list,sigma_off_list,rho_on_list) for i=1:batchsize]...)

mse_tele = Flux.mse(solution_tele,check_sol)

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
        grads = gradient(()->loss_func(p1) , ps)
        Flux.update!(opt, ps, grads)

        mse_bd = loss_func(p1)

        solution_tele = hcat([solve_tele(ps_list[i][1],ps_list[i][2],ps_list[i][3]) for i=1:batchsize]...)
        mse_tele = Flux.mse(solution_tele,check_sol)

        if mse<mse_min[1]
            df = DataFrame(p1 = p1)
            CSV.write("Topology/params_trained_bp_tele.csv",df)
            mse_min[1] = mse
        end
        print(mse,"\n")
    end
end

using CSV,DataFrames
df = CSV.read("Topology/params_trained_bp.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);
# end

solution = hcat([sol(p1,ρ_list[i],P_0_list[i]) for i=1:batchsize]...)
mse = Flux.mse(solution,train_sol)

plot(0:N-1,solution,linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,train_sol,linewidth = 3,label="exact",title="steady-state",line=:dash)

function plot_distribution(set)
    plot(0:N-1,solution[:,set],linewidth = 3,label="NN-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol[:,set],linewidth = 3,label="exact",title=join(["ρ=",ρ_list[set]]),line=:dash)
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

for i = 1:5
    p = plot_channel(i)
    savefig(p,"Topology/train_results/fig_$i.svg")
end