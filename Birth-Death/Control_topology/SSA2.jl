using Distributions,Plots,DelimitedFiles
using CSV,DataFrames,StatsBase
using Catalyst
using DelaySSAToolkit

rn = @reaction_network begin
    ρ, 0 --> N
end ρ

jumpsys = convert(JumpSystem, rn, combinatoric_ratelaws = false)

u0 = [0.]
tf = 200
saveat = 1
de_chan0 = [[]]
ρ = 0.0282*3.46
p = [ρ]
tspan = (0.0, tf)
# aggregatoralgo = DelayRejection()
# aggregatoralgo = DelayMNRM()
aggregatoralgo = DelayDirect()
# aggregatoralgo = DelayDirectCR()
dprob = DiscreteProblem(u0, tspan, p)

τ = 120.0
delay_trigger_affect! = function (integrator, rng)
    append!(integrator.de_chan[1], τ)
end
delay_trigger = Dict(1 => delay_trigger_affect!)
delay_complete = Dict(1 => [1 => -1])
delay_interrupt = Dict()
delaysets = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)
djprob = DelayJumpProblem(jumpsys, dprob, aggregatoralgo, delaysets, de_chan0,
                          save_positions = (false, false), save_delay_channel = false)
seed = 3
sol = @time solve(djprob, SSAStepper(), seed = seed, saveat = 1)


Sample_size = Int(1e4)
ens_prob = EnsembleProblem(djprob)
ens = solve(ens_prob, SSAStepper(), EnsembleThreads(), trajectories = Sample_size,
            saveat = 1)

using Catalyst.EnsembleAnalysis
tmax = Int(tf+1)
solnet = zeros(tmax,Sample_size)
for i =1:tmax
    solnet[i,:] = componentwise_vectors_timepoint(ens, Int(i-1))[1]
end
solnet

N = 65
train_sol = zeros(N+1,size(solnet,1))
for i =1:size(solnet,1)
    probability = convert_histo(vec(solnet[i,:]))[2]
    if length(probability)<N+1
        train_sol[1:length(probability),i] = probability
    else
        train_sol[1:N+1,i] = probability[1:N+1]
    end
end
train_sol
plot(0:N,train_sol[:,5])

using StatsBase,Plots
mean_SSA_X = [mean(solnet[i,:]) for i=1:size(solnet,1)]
plot(mean_SSA_X[1:tmax],label="X",linewidth = 3,legend=:bottomright)
