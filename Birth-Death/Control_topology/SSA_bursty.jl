## This example shows an application of `DelaySSA.jl` for a bursty model with delay

## A Bursty model with delay is described as
# ab^n/(1+b)^(n+1): 0 -> n P, which triggers n P to degrade after delay time τ

using DelaySSAToolkit
using Catalyst
begin # construct reaction network
    @parameters a b t
    @variables X(t)
    burst_sup = 30
    rxs = [
        Reaction(a * b^i / (1 + b)^(i + 1), nothing, [X], nothing, [i]) for i in 1:burst_sup
    ]
    rxs = vcat(rxs)
    @named rs = ReactionSystem(rxs, t, [X], [a, b])
end

# convert the ReactionSystem to a JumpSystem
jumpsys = convert(JumpSystem, rs; combinatoric_ratelaws=false)

u0 = [0]
de_chan0 = [[]]
tf = 200.0
tspan = (0, tf)
ps = [0.0282, 3.46]
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
# alg = DelayRejection()
alg = DelayCoevolve()

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
@time ens = solve(ensprob, SSAStepper(), EnsembleSerial(); 
                trajectories=Sample_size,saveat=1)

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


N = 100
train_sol_end = zeros(N+1)
sol_end = componentwise_vectors_timepoint(ens, tf)[1]

probability = convert_histo(vec(sol_end))[2]
if length(probability)<N+1
    train_sol_end[1:length(probability)] = probability
else
    train_sol_end[1:N+1] = probability[1:N+1]
end
train_sol_end
plot(train_sol_end,lw=3,title=join(["i=",burst_sup]))
