using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots,CSV,DataFrames
using BlackBoxOptim

include("../../../utils.jl")

function bursty(N,a,b,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

N = 100
a = 0.0282
b = 3.46
τ = 120
temp1 = bursty(N,a,b,τ)

Ex = a*b*τ
Dx = a*b*τ+2a*b^2*τ

a = 2Ex^2/(Dx-Ex)τ
b = (Dx-Ex)/2Ex


P2mean(temp1)
P2var(temp1)

exp(-a*b*τ/(1+b))