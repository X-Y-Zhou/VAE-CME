using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots

include("../../../utils.jl")

#exact solution
function bursty(N,a,b,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

function f_NN(x,l,m,n,o)
    return l*x^m/(n+x^o)
end

a = 0.0282;
b = 3.46;
τ = 120;

N = 64
train_sol = bursty(N,a,b,τ)

a_list = [0.0082,0.0282]
b_list = [1.46,3.46]
# τ_list = [100,110,120,130,140]
l_ablist = length(a_list)*length(b_list)

ab_list = [[a_list[i],b_list[j]] for i=1:length(a_list) for j=1:length(b_list)]
# abτ_list = [[a_list[i],b_list[j],τ_list[k]] for i=1:length(a_list) for j=1:length(b_list) for k=1:length(τ_list)]

train_sol = [bursty(N,ab_list[i][1],ab_list[i][2],τ) for i=1:l_ablist]
# train_sol = [bursty(N,abτ_list[i][1],abτ_list[i][2],abτ_list[i][3]) for i=1:length(abτ_list)]


# model initialization
latent_size = 5;
encoder = Chain(Dense(N, 200,tanh),Dense(200, latent_size * 2));
decoder = Chain(Dense(latent_size, 200),Dense(200 , 4),x ->exp.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

params1
params2

# p1 = params1
# p2 = params2
# x = P_0_list[1]

# h = re1(p1)(x)
# μ, logσ = split_encoder_result(h, latent_size)
# z = reparameterize.(μ, logσ, ϵ)
# l,m,n,o = re2(p2)(z)
# NN = f_NN.(1:N-1,l,m,n,o)

#CME
function f1!(x,p1,p2,a,b,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    l,m,n,o = re2(p2)(z)
    NN = f_NN.(1:N-1,l,m,n,o)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

#solve P
P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
P_0_list = [[pdf(NegativeBinomial(ab_list[i][1]*τ, 1/(1+ab_list[i][2])),j) for j=0:N-1] for i=1:l_ablist]

ϵ = zeros(latent_size)
sol(p1,p2,a,b,ϵ,P0) = nlsolve(x->f1!(x,p1,p2,a,b,ϵ),P0).zero
# sol(params1,params2,a,b,ϵ,P_0_list[25])

function loss_func(p1,p2,ϵ)
    sol_cme = [sol(p1,p2,ab_list[i][1],ab_list[i][2],ϵ,P_0_list[i]) for i=1:l_ablist]
        
    mse = sum(Flux.mse(sol_cme[i],train_sol[i]) for i=1:l_ablist)/l_ablist
    print(mse," ")

    μ_logσ_list = [split_encoder_result(re1(p1)(sol_cme[i]), latent_size) for i=1:l_ablist]
    kl = sum([(0.5f0 * sum(exp.(2f0 .* μ_logσ_list[i][2]) + μ_logσ_list[i][1].^2 .- 1 .- (2 .* μ_logσ_list[i][2])))  
        for i=1:l_ablist])/l_ablist
    print(kl," ")

    loss = λ*mse + kl
    print(loss,"\n")
    return loss
end

λ = 50000000

#check λ if is appropriate
ϵ = zeros(latent_size)
loss_func(params1,params2,ϵ)
@time grads = gradient(()->loss_func(params1,params2,ϵ) , ps)

epochs_all = 0

lr_list = [0.01,0.008,0.006,0.004,0.002,0.001]
lr_list = [0.005,0.0025,0.0015,0.001]
lr_list = [0.006,0.004,0.002,0.001]


using CSV,DataFrames
df = CSV.read("Bursty/Control_rate_Inference/control_kinetic/params_ck5-2.csv",DataFrame)
params1 = df.params1
params2 = df.params2[1:length(params2)]
ps = Flux.params(params1,params2);

# training
lr = 0.001;  #lr需要操作一下的

# for lr in lr_list
opt= ADAM(lr);
epochs = 20
epochs_all = epochs_all + epochs
print("learning rate = ",lr)
mse_list = []

@time for epoch in 1:epochs
    ϵ = rand(Normal(),latent_size)
    print(epoch,"\n")
    grads = gradient(()->loss_func(params1,params2,ϵ) , ps)
    Flux.update!(opt, ps, grads)

    ϵ = zeros(latent_size)
    solution = [sol(params1,params2,ab_list[i][1],ab_list[i][2],ϵ,P_0_list[i]) for i=1:l_ablist]
    mse = sum(Flux.mse(solution[i],train_sol[i]) for i=1:l_ablist)/l_ablist

    if mse<mse_min[1]
        df = DataFrame( params1 = params1,params2 = vcat(params2,[0 for i=1:length(params1)-length(params2)]))
        CSV.write("Bursty/Control_rate_Inference/control_kinetic/params_ck5-2.csv",df)
        mse_min[1] = mse
    end

    push!(mse_list,mse)
    print(mse,"\n")
end

params1
params2

mse_min = [0.0072110871408849865]
mse_min 

ϵ = zeros(latent_size)
solution = [sol(params1,params2,ab_list[i][1],ab_list[i][2],ϵ,P_0_list[i]) for i=1:l_ablist]
mse = sum(Flux.mse(solution[i],train_sol[i]) for i=1:l_ablist)/l_ablist

function plot_distribution(set)
    plot(0:N-1,solution[set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,train_sol[set],linewidth = 3,label="exact",title=join(["a,b,τ=",ab_list[set]]),line=:dash)
end

function plot_all()
    p1 = plot_distribution(1)
    p2 = plot_distribution(2)
    p3 = plot_distribution(3)
    p4 = plot_distribution(4)
    plot(p1,p2,p3,p4,size=(600,600),layout=(2,2))
end
plot_all()
# savefig("Bursty/Control_rate_Inference/control_kinetic/fitting.svg")


a_list_pre = [0.0082,0.0132,0.0182,0.0232,0.0282]
b_list_pre = [1.46,1.96,2.46,2.96,3.46]
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
    solution = sol(params1,params2,a,b,ϵ,P_0)
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
    p25 = plot_distribution(25)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
         p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,1500),layout=(5,5))
end
plot_all()
savefig("Bursty/Control_rate_Inference/control_kinetic/predicting.svg")
