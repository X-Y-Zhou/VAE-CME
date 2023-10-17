using Flux
using DiffEqFlux
using NLsolve
using Zygote
using Zygote: @adjoint
using IterativeSolvers
using LinearMaps
#using SparseArrays
using LinearAlgebra
using TaylorSeries
using Plots: plot, plot!

## Generating function Taylor expansion
function steady_prob(param)
    τ = param[3]
    NT = Int(param[4])
    g(u) = exp(param[2]*param[1]*u/(1-param[1]*u)*τ)
    taylor_gen = taylor_expand(u->g(u),-1,order=NT)
    p_ge = zeros(Float32,NT)
    for i = 1 : NT
        p_ge[i] = taylor_gen[i-1]
    end
    return p_ge
end

## Data generation
param = Float32[3.46, 0.0282, 120, 120]
NT = Int(param[4])

data = steady_prob(param)
plot(data)

data = data/sum(data)

## Define NLsolve adjoint
@adjoint nlsolve(f, x0; kwargs...) =
    let result = nlsolve(f, x0; kwargs...)
        result, function(vresult)
            dx = vresult[].zero
            x = result.zero
            _, back_x = Zygote.pullback(f, x)

            JT(df) = back_x(df)[1]
            # solve JT*df = -dx
            L = LinearMap(JT, length(x0))
            df = gmres(L,-dx)

            _, back_f = Zygote.pullback(f -> f(x), f)
            return (back_f(df)[1], nothing, nothing)
        end
    end

## Define neural network propensity
m = Chain(Dense(NT+1, 5, tanh),Dense(5, 1))
NNet = SkipConnection(m, (mx,x)->relu(0.30f0*mx[1]+x[end]/param[3]))
p1, re = Flux.destructure(NNet)
ps = Flux.params(p1)

## Define nonlinear governing equations
function Delay_eq(x,p,param)
    b = param[1]
    ρ = param[2]
    N = Int(param[4])-1
    NN = [re(p)(vcat(x,i))[1] for i in 1:N]

    return vcat(-ρ*b/(1+b)*x[1] + NN[1]*x[2],
    [sum(ρ*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - (ρ*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N],
    sum(x)-1.0f0)
end

f(x,p) = Delay_eq(x,p,param)
x = data

solve_x(p1) = nlsolve(x -> f(x, p1), data, ftol=1e-10,method= :anderson,m=5).zero
obj(p1) = sum(abs2,solve_x(p1)-data)
opt = ADAM(0.001)

for epoch = 1 : 50
    grads = gradient(() -> obj(p1), ps)
    Flux.update!(opt,ps,grads)
    #print(p1[1])
    evalcb() = @show(epoch,obj(p1))
    evalcb()
end

solution = solve_x(p1)
Flux.mse(solution,data)

plot(solve_x(p1))
plot!(data)

param = Float32[3.46, 0.0282, 120, 120]
data = steady_prob(param)
# x = data
solution = solve_x(p1)
Flux.mse(data,solution)

param = Float32[1.06, 0.0182, 120, 120]
data = steady_prob(params1)
# x = data
solution = solve_x(p1)
Flux.mse(data,solution)

plot(solution,lw=3,label="NN-CME")
plot!(data,lw=3,label="SSA",line=:dash)


df = DataFrame(p1 = p1)
CSV.write("Bursty/Control_rate_Inference/control_kinetic/params_ML2.csv",df)

using CSV,DataFrames
df = CSV.read("Bursty/Control_rate_Inference/control_kinetic/params_ML2.csv",DataFrame)
p1 = df.p1
ps = Flux.params(p1);

N = 120
a_list_pre = [0.0082,0.0132,0.0182,0.0232,0.0282]
b_list_pre = [1.46,1.96,2.46,2.96,3.46]
l_ablist_pre = length(a_list_pre)*length(b_list_pre)

ab_list_pre = [[a_list_pre[i],b_list_pre[j]] for i=1:length(a_list_pre) for j=1:length(b_list_pre)]

τ = 120
train_sol = [bursty(N,ab_list_pre[i][1],ab_list_pre[i][2],τ) for i=1:l_ablist_pre]

solution_list = []
for i=1:l_ablist_pre
    print(i,"\n")
    a = ab_list_pre[i][1]
    b = ab_list_pre[i][2]
    param = Float32[b, a, 120, 120]

    # data = steady_prob(param)
    solution = solve_x(p1)
    push!(solution_list,solution)
end
solution_list
solution_list[24]
solution_list[25]

solution_list
solution_list[24]
solution_list[25]

function  plot_distribution(set)
    p=plot(0:N-1,solution_list[set],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N-1,train_sol[set],linewidth = 3,label="exact",line=:dash,title=join(["ab=",ab_list_pre[set]]))
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