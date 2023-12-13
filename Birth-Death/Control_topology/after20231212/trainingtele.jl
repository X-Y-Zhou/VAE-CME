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

# model initialization
model = Chain(Dense(N, 10, tanh), Dense(10, N-1), x -> 0.03.*x .+ [i/τ for i in 1:N-1], x -> relu.(x));
p1, re = Flux.destructure(model);
ps = Flux.params(p1);
p1

#CME
function f1!(x,p,sigma_on,sigma_off,rho_on)
    NN1 = re(p)(x[1:N])
    NN2 = re(p)(x[N+1:2*N])
    # l,m,n,o = re(p)(z)
    # NN = f_NN.(1:N-1,l,m,n,o)
    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)
end

#solve P
P_0_distribution = Poisson(rho_on*τ*sigma_on)
P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]

sol(p,sigma_on,sigma_off,rho_on,P_0) = nlsolve(x->f1!(x,p,sigma_on,sigma_off,rho_on),P_0).zero
solution = sol(p1,sigma_on,sigma_off,rho_on,P_0_split)
solution = solution[1:N]+solution[N+1:2*N]

function loss_func(p)
    sol_cme = sol(p,sigma_on,sigma_off,rho_on,P_0_split)
    sol_cme = sol_cme[1:N]+sol_cme[N+1:2*N]
    mse = Flux.mse(sol_cme,train_sol)
    loss = mse
    return loss
end

@time loss_func(p1)
@time grads = gradient(()->loss_func(p1) , ps)

lr = 0.006;  #lr需要操作一下的

# for lr in lr_list
using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231212/params_trained_tele.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

# # training

opt= ADAM(lr);
epochs = 30
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    print(epoch,"\n")
    grads = gradient(()->loss_func(p1) , ps)
    Flux.update!(opt, ps, grads)

    mse = loss_func(p1)
    if mse<mse_min[1]
        df = DataFrame(p1 = p1)
        CSV.write("Birth-Death/Control_topology/after20231212/params_trained_tele.csv",df)
        mse_min[1] = mse
    end
    
    push!(mse_list,mse)
    print(mse,"\n")
end


mse_min = [0.00010064636802613852]
mse_min 

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231212/params_trained_tele.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

solution = sol(p1,sigma_on,sigma_off,rho_on,P_0_split)
solution = solution[1:N]+solution[N+1:2*N]
mse = Flux.mse(solution,train_sol)

plot(0:N-1,solution,linewidth = 3,label="NN-CME",xlabel = "# of products \n", ylabel = "\n Probability")
plot!(0:N-1,train_sol,linewidth = 3,label="exact",line=:dash)

function plot_distribution(set)
    plot(0:N-1,solution[set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol[set],linewidth = 3,label="exact",title=join(["a,b,τ=",ab_list[set]]),line=:dash)
end
plot_distribution(1)

function plot_all()
    p1 = plot_distribution(1)
    p2 = plot_distribution(2)
    p3 = plot_distribution(3)
    p4 = plot_distribution(4)
    plot(p1,p2,p3,p4,size=(600,600),layout=(2,2))
end
plot_all()
# # savefig("Control_rate_Inference/control_kinetic/fitting.svg")

function plot_all()
    p1 = plot_distribution(1)
    p2 = plot_distribution(2)
    p3 = plot_distribution(3)
    p4 = plot_distribution(4)
    p5 = plot_distribution(5)
    p6 = plot_distribution(6)
    p7 = plot_distribution(7)
    p8 = plot_distribution(8)
    p9 = plot_distribution(9)
    p10 = plot_distribution(10)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,size=(1500,600),layout=(2,5))
end
plot_all()

a_list = [0.04,0.1]
b_list = [2.,3.]


a_list_pre = [0.04,0.06,0.08,0.1]
b_list_pre = [2,2.2,2.4,2.6,2.8,3]
l_ablist_pre = length(a_list_pre)*length(b_list_pre)

ab_list_pre = [[a_list_pre[i],b_list_pre[j]] for i=1:length(a_list_pre) for j=1:length(b_list_pre)]

solution_list = []
for i=1:l_ablist_pre
    print(i,"\n")
    a = ab_list_pre[i][1]
    b = ab_list_pre[i][2]
    P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]

    ϵ = zeros(latent_size)
    solution = sol(p1,a,b,P_0)
    push!(solution_list,solution)
end

function  plot_distribution(set)
    p=plot(0:N-1,solution_list[set],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N-1,bursty(N,ab_list_pre[set][1],ab_list_pre[set][2],τ),linewidth = 3,label="exact",line=:dash,title=join(["ab=",ab_list_pre[set]]))
end

function plot_all()
    p1 = plot_distribution(1)
    p2 = plot_distribution(2)
    p3 = plot_distribution(3)
    p4 = plot_distribution(4)
    p5 = plot_distribution(5)
    p6 = plot_distribution(6)
    p7 = plot_distribution(7)
    p8 = plot_distribution(8)
    p9 = plot_distribution(9)
    p10 = plot_distribution(10)
    p11 = plot_distribution(11)
    p12 = plot_distribution(12)
    p13 = plot_distribution(13)
    p14 = plot_distribution(14)
    p15 = plot_distribution(15)
    p16 = plot_distribution(16)
    p17 = plot_distribution(17)
    p18 = plot_distribution(18)
    p19 = plot_distribution(19)
    p20 = plot_distribution(20)
    p21 = plot_distribution(21)
    p22 = plot_distribution(22)
    p23 = plot_distribution(23)
    p24 = plot_distribution(24)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
         p16,p17,p18,p19,p20,p21,p22,p23,p24,size=(1800,1200),layout=(4,6))
end
plot_all()
savefig("Control_rate_Inference/control_kinetic/predicting.pdf")
