using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

# check 
train_solnet = readdlm("Birth-Death/Control_topology/after20231210/train_data_ssa2.csv", ',')[2:end,:]
ρ_listnet = readdlm("Birth-Death/Control_topology/after20231210/p.csv", ',')[2:end,:]

ab_list = []
for i = 1:6
N = 70
τ = 120
train_sol = train_solnet[:,i][1:N]
ρ_list = vcat(ρ_listnet[:,i],zeros(N-length(ρ_listnet[:,i])))
ρ_list

Ex = P2mean(train_sol)
Dx = P2var(train_sol)

a = 2Ex^2/(Dx-Ex)τ
b = (Dx-Ex)/2Ex

push!(ab_list,[a,b])
end
ab_list

i = 5
ρ_list = vcat(ρ_listnet[:,i],zeros(N-length(ρ_listnet[:,i])))
train_sol = train_solnet[:,i][1:N]
ab_list[i]
ab_list
# train_solnet[:,i][1:N]

# model initialization
latent_size = 2;
encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
decoder = Chain(Dense(latent_size, 10),Dense(10 , N-1),x-> 0.03.*x.+[i/τ  for i in 1:N-1],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231210/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

#CME
function f1!(x,p1,p2,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN = re2(p2)(z)

    return vcat(-sum(ρ_list)*x[1]+NN[1]*x[2],[sum(ρ_list[i-j]*x[j] for j in 1:i-1) - 
            (sum(ρ_list)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

#solve P
P_0_distribution = Poisson(mean(ρ_list[1:10]*120))
# P_0_distribution = NegativeBinomial(a*τ, 1/(1+b))
P_0 = [pdf(P_0_distribution,i) for i=0:N-1]

ϵ = zeros(latent_size)
sol(p1,p2,ϵ,P_0) = nlsolve(x->f1!(x,p1,p2,ϵ),P_0).zero

solution = sol(params1,params2,ϵ,P_0)
mse = Flux.mse(solution,train_sol)

plot(0:N-1,solution,linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,train_sol,linewidth = 3,label="SSA",title="steady-state",line=:dash)


# topo bursty
# exact
function bursty(N,a,b,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

Ex = P2mean(train_sol)
Dx = P2var(train_sol)

a = 2Ex^2/(Dx-Ex)τ
b = (Dx-Ex)/2Ex

a_list = [0.04,0.06,0.07,0.08,0.1]
b_list = [2,2.25,2.4,2.5,2.75]
ab_list = [[a_list[i],b_list[j]] for i=1:length(a_list) for j=1:length(b_list)]
N = 70

check_sol_list = []
solution_list = []

for i = 1:length(ab_list)
    print(i,"\n")
    check_sol = bursty(N,ab_list[i][1],ab_list[i][2],τ)
    solution = solve_bursty(ab_list[i][1],ab_list[i][2])
    push!(check_sol_list,check_sol)
    push!(solution_list,solution)
end

function  plot_distribution(set)
    p=plot(0:N-1,solution_list[set],linewidth = 3,label="topo",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N-1,check_sol_list[set],linewidth = 3,label="exact",line=:dash,title=join(["ab=",ab_list[set]]))
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
    p25 = plot_distribution(25)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
         p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,1500),layout=(5,5))
end
plot_all()



mse = Flux.mse(solution,check_sol)

plot(0:N-1,solution,linewidth = 3,label="topo-bursty",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,check_sol,linewidth = 3,label="bursty-exact",title="steady-state",line=:dash)

# model
latent_size = 2;
encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
decoder = Chain(Dense(latent_size, 10),Dense(10 , N-1),x-> 0.03.*x.+[i/τ  for i in 1:N-1],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231210/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

function solve_bursty(a,b)
function f1!(x,p1,p2,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN = re2(p2)(z)

    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
    (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
P_0 = [pdf(P_0_distribution,i) for i=0:N-1]

ϵ = zeros(latent_size)
sol(p1,p2,ϵ,P_0) = nlsolve(x->f1!(x,p1,p2,ϵ),P_0).zero

solution = sol(params1,params2,ϵ,P_0)
return solution
end

mse = Flux.mse(solution,check_sol)

plot(0:N-1,solution,linewidth = 3,label="topo-bursty",xlabel = "# of products", ylabel = "\n Probability")
plot!(0:N-1,check_sol,linewidth = 3,label="bursty-exact",title="steady-state",line=:dash)


# topo tele
sigma_on = a
sigma_off = 1
rho_on = b
rho_off = 0.0
gamma= 0.0

a = sigma_on
b = rho_on/sigma_off
N = 70

# check_sol = vec(readdlm("Birth-Death/Control_topology/after20231210/ssa_tele.csv", ',')[2:N+1,:])

a_list = [0.04,0.06,0.07,0.08,0.1]
b_list = [2,2.25,2.4,2.5,2.75]
ab_list = [[a_list[i],1.,b_list[j]] for i=1:length(a_list) for j=1:length(b_list)]
N = 70

solution_list = []

for i = 1:length(ab_list)
    print(i,"\n")
    solution = solve_tele(ab_list[i][1],ab_list[i][2],ab_list[i][3])
    push!(solution_list,solution)
end

function  plot_distribution(set)
    p=plot(0:N-1,solution_list[set],linewidth = 3,label="topo",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N-1,train_sol_end_list[set],linewidth = 3,label="exact",line=:dash,title=join(["ab=",ab_list[set]]))
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
    p25 = plot_distribution(25)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
         p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,1500),layout=(5,5))
end
plot_all()

# model
latent_size = 2;
encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
decoder = Chain(Dense(latent_size, 10),Dense(10 , N-1),x-> 0.03.*x.+[i/τ  for i in 1:N-1],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

using CSV,DataFrames
df = CSV.read("Birth-Death/Control_topology/after20231210/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

# P0,P1
function solve_tele(sigma_on,sigma_off,rho_on)
function f1!(x,p1,p2,ϵ)
    h = re1(p1)(x[1:N])
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN1 = re2(p2)(z)

    h = re1(p1)(x[N+1:2*N])
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN2 = re2(p2)(z)

    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)
end

P_0_distribution = Poisson(rho_on*τ*sigma_on)
P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]

ϵ = zeros(latent_size)
sol(p1,p2,ϵ,P_0) = nlsolve(x->f1!(x,p1,p2,ϵ),P_0).zero
solution = sol(params1,params2,ϵ,P_0_split)
solution = solution[1:N]+solution[N+1:2*N]
return solution
end

# mse = Flux.mse(solution,check_sol)

plot(0:N-1,solution,linewidth = 3,label="topo-tele",xlabel = "# of products \n", ylabel = "\n Probability")
plot!(0:N-1,check_sol,linewidth = 3,label="SSA",line=:dash)



function  plot_distribution(set)
    p=plot(0:N-1,bursty(N,ab_list[set][1],ab_list[set][2],75),linewidth = 3,label="exact",title=join(["ab=",ab_list[set]]))
end
set = 1
plot_distribution(set)

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
    p25 = plot_distribution(25)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
         p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,1500),layout=(5,5))
end
plot_all()

