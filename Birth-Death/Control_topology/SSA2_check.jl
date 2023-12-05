## This example shows an application of `DelaySSA.jl` for a bursty model with delay

## A Bursty model with delay is described as
# ab^n/(1+b)^(n+1): 0 -> n P, which triggers n P to degrade after delay time τ

using DelaySSAToolkit
using Catalyst
include("../../utils.jl")

begin # construct reaction network
    @parameters ρ t
    @variables X(t)
    burst_sup = 10
    rxs = [
        Reaction(ρ/i, nothing, [X], nothing, [i]) for i in 1:burst_sup
    ]
    # rxs = [
    #     Reaction(ρ/i, [G], [G,X], [1], [1,i]) for i in 1:burst_sup
    # ]
    rxs = vcat(rxs)
    @named rs = ReactionSystem(rxs, t, [X], [ρ])
end

# convert the ReactionSystem to a JumpSystem
jumpsys = convert(JumpSystem, rs; combinatoric_ratelaws=false)

u0 = [0]
de_chan0 = [[]]
tf = 200.0
tspan = (0, tf)
# ps = [0.0282, 3.46]
ps = [0.0083]
dprob = DiscreteProblem(jumpsys, u0, tspan, ps)
τ = 120.0

delay_trigger_affect! = []
for i in 1:burst_sup
    push!(delay_trigger_affect!, function (integrator, rng)
        append!(integrator.de_chan[1], fill(τ, i))
    end)
end

delay_trigger_affect!
delay_trigger = Dict([Pair(i, delay_trigger_affect![i]) for i in 1:burst_sup])
delay_complete = Dict(1 => [1 => -1])
delay_interrupt = Dict()
alg = DelayRejection()
# alg = DelayCoevolve()

delayjumpset = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)
jprob = DelayJumpProblem(
    jumpsys,
    dprob,
    alg,
    delayjumpset,
    de_chan0;
    save_positions=(true, true),
    save_delay_channel=false,
)
# saveat = 0:1:tf
seed = 4
@time sol = solve(jprob, SSAStepper(); seed=seed)

Sample_size = Int(1e5)
ensprob = EnsembleProblem(jprob)
@time ens = solve(ensprob, SSAStepper(), EnsembleSerial(); trajectories=Sample_size,saveat=1)

using Catalyst.EnsembleAnalysis
tmax = Int(tf+1)
solnet = zeros(tmax,Sample_size)
for i =1:tmax
    solnet[i,:] = componentwise_vectors_timepoint(ens, Int(i-1))[1]
end
solnet

using StatsBase,Plots
mean_SSA_X = [mean(solnet[i,:]) for i=1:size(solnet,1)]
plot(mean_SSA_X[1:tmax],label="X",linewidth = 3,legend=:bottomright)


N = 200
train_sol_end = zeros(N+1)
sol_end = componentwise_vectors_timepoint(ens, tf)[1]

probability = convert_histo(vec(sol_end))[2]
if length(probability)<N+1
    train_sol_end[1:length(probability)] = probability
else
    train_sol_end[1:N+1] = probability[1:N+1]
end
train_sol_end
plot(train_sol_end,title=join(["i=",burst_sup]))

reshape(train_sol_end,N+1,1)

df = DataFrame(reshape(train_sol_end,N+1,1),"x1")
CSV.write("Birth-Death/Inference/data/set$set/$(μ)-sqrt($(round(σ^2))).csv",df)

temp = reshape(train_sol_end,N+1,1)
df = DataFrame(reshape(train_sol_end,N+1,1),:auto)
CSV.write("Control_topology/temp.csv",df)


temp = [1. 0.]
temp = reshape(temp,2,1)
DataFrame(temp, :auto)


# Check with the exact probability distribution
using TaylorSeries
function taylor_coefficients(NT::Int, at_x, gen::Function)
    Q = zeros(NT)
    taylor_gen = taylor_expand(u -> gen(u), at_x; order=NT)
    for j in 1:NT
        Q[j] = taylor_gen[j - 1]
    end
    return Q
end
function delay_bursty(params, NT::Int)
    a, b, τ, t = params
    gen1(u) = exp(a * b * min(τ, t) * u / (1 - b * u))
    return taylor_coefficients(NT, -1, gen1)
end

using Catalyst.EnsembleAnalysis
using Plots;
theme(:vibrant);
sol_end = componentwise_vectors_timepoint(ens, tf)
histogram(
    sol_end;
    bins=0:1:60,
    normalize=:pdf,
    label="Delay SSA",
    fillalpha=0.6,
    linecolor=:orange,
)

sol_exact = delay_bursty([ps; 130; 200], 61)
fig = plot!(
    0:60,
    sol_exact;
    linewidth=3,
    label="Exact solution",
    fmt=:svg,
    xlabel="# of products",
    ylabel="Probability",
)
# savefig(fig, "docs/src/assets/bursty.svg")
