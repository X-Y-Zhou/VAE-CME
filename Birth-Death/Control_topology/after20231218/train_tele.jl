using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

train_sol = vec(readdlm("Birth-Death/Control_topology/after20231212/ssa_tele.csv", ',')[2:N+1,:])
sigma_on = 0.003
sigma_off = 0.004
rho_on = 0.3
rho_off = 0.0
gamma= 0.0
τ = 120
N = 100

# model initialization
model1 = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> 0.03.*x .+ [i/τ for i in 1:N-1], x -> relu.(x));
model2 = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> 0.03.*x .+ [i/τ for i in 1:N-1], x -> relu.(x));
params1, re = Flux.destructure(model1);
params2, re = Flux.destructure(model2);
ps = Flux.params(params1,params2);

#CME
function f1!(x,p1,p2,sigma_on,sigma_off,rho_on)
    NN1 = re(p1)(x[1:N])
    NN2 = re(p2)(x[N+1:2*N])
    push!(NN1_list,NN1)
    push!(NN2_list,NN2)
    push!(x_list,x)
    # l,m,n,o = re(p)(z)
    # NN = f_NN.(1:N-1,l,m,n,o)
    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                # sum(x[1:N])-sigma_off/(sigma_on+sigma_off),

                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1,
                # sum(x[N+1:2*N])-sigma_on/(sigma_on+sigma_off)
                )
end

#solve P
P_0_distribution = Poisson(rho_on*τ*sigma_on)
P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
P_0_split = [P_0*sigma_off/(sigma_on+sigma_off);P_0*sigma_on/(sigma_on+sigma_off)]

solving(p1,p2,sigma_on,sigma_off,rho_on,P_0) = nlsolve(x->f1!(x,p1,p2,sigma_on,sigma_off,rho_on),P_0).zero
solution = solving(params1,params2,sigma_on,sigma_off,rho_on,P_0_split)
solution = solution[1:N]+solution[N+1:2*N]

function loss_func(p1,p2)
    sol_cme = solving(p1,p2,sigma_on,sigma_off,rho_on,P_0_split)
    sol_cme = sol_cme[1:N]+sol_cme[N+1:2*N]
    mse = Flux.mse(sol_cme,train_sol)
    loss = mse
    return loss
end

@time loss_func(params1,params2)
@time grads = gradient(()->loss_func(params1,params2) , ps)

lr = 0.01;  #lr需要操作一下的

# for lr in lr_list
using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231218/params_trained_tele-2.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

# # training

opt= ADAM(lr);
epochs = 30
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    print(epoch,"\n")
    grads = gradient(()->loss_func(params1,params2) , ps)
    Flux.update!(opt, ps, grads)

    mse = loss_func(params1,params2)
    if mse<mse_min[1]
        df = DataFrame(params1=params1,params2 = params2)
        CSV.write("Birth-Death/Control_topology/after20231218/params_trained_tele-2.csv",df)
        mse_min[1] = mse
    end
    
    push!(mse_list,mse)
    print(mse,"\n")
end


mse_min = [9.896340261199816e-5]
mse_min 

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231218/params_trained_tele.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

NN1_list = []
NN2_list = []
x_list = []

solution = solving(params1,params2,sigma_on,sigma_off,rho_on,P_0_split)
solution = solution[1:N]+solution[N+1:2*N]
mse = Flux.mse(solution,train_sol)

plot(0:N-1,solution,linewidth = 3,label="NN-CME",xlabel = "# of products \n", ylabel = "\n Probability")
plot!(0:N-1,train_sol,linewidth = 3,label="exact",line=:dash)


x_list[end][1:N]
x_list[end][N+1:2*N]
plot(x_list[end])
plot(x_list[end][1:N],label="P0")
plot(x_list[end][N+1:2*N],label="P1")
plot(x_list[end][1:N]+x_list[end][N+1:2*N])
x_temp = x_list[end][1:N]+x_list[end][N+1:2*N]
plot([x_temp*sigma_on/(sigma_on+sigma_off);x_temp*sigma_off/(sigma_on+sigma_off)])

NN1_list[end]
plot(NN1_list[end],label="NN1")
plot(NN2_list[end],label="NN2")
plot(NN1_list[end].+NN2_list[end])

P_split = [P0;P1]
P_split = x_list[end]
plot(P_split)

function f(NN)
    NN1 = NN[1:N-1]
    NN2 = NN[N:2*N-2]
    [(-sigma_on-rho_off)*P_split[1] + (-gamma+NN1[1])*P_split[2] + sigma_off*P_split[N+1];
    [rho_off*P_split[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*P_split[i] + (-i*gamma+NN1[i])*P_split[i+1] + sigma_off*P_split[i+N] for i in 2:N-1];
    rho_off*P_split[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*P_split[N] + sigma_off*P_split[2*N];
    # sum(P_split[1:N])-sigma_off/(sigma_on+sigma_off)
    
    sigma_on*P_split[1] + (-sigma_off-rho_on)*P_split[N+1] + (-gamma+NN2[1])*P_split[N+2];
    [sigma_on*P_split[i-N] + rho_on*P_split[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*P_split[i] + (-(i-N)*gamma+NN2[i-N])*P_split[i+1] for i in (N+2):(2*N-1)];
    # sigma_on*P_split[N] + rho_on*P_split[2*N-1]+(-sigma_off+N*gamma -NN2[N-1])*P_split[2*N]
    # sum(P_split[N+1:2*N])-sigma_on/(sigma_on+sigma_off)
    ]
end

solution = nlsolve(f, [1.;zeros(2*N-2)])
NN_solution = solution.zero
plot(NN_solution[1:N-1],label="NN1-compute",lw=3)
plot!(NN1_list[end],label="NN1",lw=3,line=:dash)

plot(NN_solution[N:2*N-2],label="NN2-compute",lw=3)
plot!(NN2_list[end],label="NN2",lw=3,line=:dash)

NN_solution = NN_solution[1:N-1]+NN_solution[N:2*N-2]
plot(NN_solution)

f([NN2_list[end];NN1_list[end]])
sum(f([NN2_list[end];NN1_list[end]]))


P_split = x_list[end]
P_split = [P0;P1]
function f1!(NN,sigma_on,sigma_off,rho_on)
    NN1 = NN[1:N-1]
    NN2 = NN[N:2*N-2]
    return vcat((-sigma_on-rho_off)*P_split[1] + (-gamma+NN1[1])*P_split[2] + sigma_off*P_split[N+1],
                [rho_off*P_split[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*P_split[i] + (-i*gamma+NN1[i])*P_split[i+1] + sigma_off*P_split[i+N] for i in 2:N-1],
                rho_off*P_split[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*P_split[N] + sigma_off*P_split[2*N],
                
                sigma_on*P_split[1] + (-sigma_off-rho_on)*P_split[N+1] + (-gamma+NN2[1])*P_split[N+2],
                [sigma_on*P_split[i-N] + rho_on*P_split[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*P_split[i] + (-(i-N)*gamma+NN2[i-N])*P_split[i+1] for i in (N+2):(2*N-1)],
                )
end

solving(sigma_on,sigma_off,rho_on,P_0) = nlsolve(NN->f1!(NN,sigma_on,sigma_off,rho_on),P_0).zero
solution = solving(sigma_on,sigma_off,rho_on,P_0_split)
solution = solution[1:N]+solution[N+1:2*N]
plot(solution)



train_sol = bursty(N,a,b,τ)
a = 0.0282
b = 3.46
function f(NN)
    [-a*b/(1+b)*train_sol[1]+NN[1]*train_sol[2];
    [sum(a*(b/(1+b))^(i-j)/(1+b)*train_sol[j] for j in 1:i-1) - 
    (a*b/(1+b)+NN[i-1])*train_sol[i] + NN[i]*train_sol[i+1] for i in 2:N-1];0]
end

sol = nlsolve(f, [2.;zeros(N-1)])
NN_solution = sol.zero
plot(NN_solution)
NN_solution = NN_bursty(N,a,b,τ)
plot!(NN_solution)
