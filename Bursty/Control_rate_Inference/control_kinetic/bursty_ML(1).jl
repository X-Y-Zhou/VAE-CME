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
NNet = Chain(Dense(NT, 1, tanh),Dense(1, NT-1),x->0.3.*x.+[i/param[3] for i in 1:NT-1], x->relu.(x))
p1, re = Flux.destructure(NNet)
ps = Flux.params(p1)

## Define nonlinear governing equations
function Delay_eq(x,p,param)
    b = param[1]
    ρ = param[2]
    N = Int(param[4])-1
    NN = re(p)(x)

    return vcat(-ρ*b/(1+b)*x[1] + NN[1]*x[2],
    [sum(ρ*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - (ρ*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N],
    sum(x)-1.0f0)
end

f(x,p) = Delay_eq(x,p,param)
x = data

solve_x(p1) = nlsolve(x -> f(x, p1), data, ftol=1e-10,method= :anderson,m=5).zero
obj(p1) = sum(abs2,solve_x(p1)-data)
opt = ADAM(0.001)

for epoch = 1 : 20
    grads = gradient(() -> obj(p1), ps)
    Flux.update!(opt,ps,grads)
    #print(p1[1])
    evalcb() = @show(epoch,obj(p1))
    evalcb()
end
plot(solve_x(p1))
plot!(data)
