using Distributions,Plots,DelimitedFiles
using CSV,DataFrames,StatsBase
using Catalyst
using DelaySSAToolkit

rn = @reaction_network begin
    ρ, 0 --> N
end ρ

jumpsys = convert(JumpSystem, rn, combinatoric_ratelaws = false)

u0 = [0.]
tf = 100
saveat = 1
de_chan0 = [[]]
ρ = 20
p = [ρ]
tspan = (0.0, tf)
# aggregatoralgo = DelayRejection()
# aggregatoralgo = DelayMNRM()
aggregatoralgo = DelayDirect()
# aggregatoralgo = DelayDirectCR()
dprob = DiscreteProblem(u0, tspan, p)

τ = 10.0
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

u = [sol.u[i][1] for i=1:length(sol.u)]
plot(u,lw=3)

using DataFrames,CSV
df = DataFrame( x1 = u)
CSV.write("Birth-Death/SSA_seed$seed.csv",df)

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

using DataFrames,CSV
df = DataFrame(solnet,:auto)
CSV.write("machine-learning/ode/birth-death/data.csv",df)

using StatsBase,Plots
mean_SSA_X = [mean(solnet[i,:]) for i=1:size(solnet,1)]
plot(mean_SSA_X[1:tmax],label="X",linewidth = 3,legend=:bottomright)

t = 8
plot(birth_death(ρ,t,300),linewidth=3,label="exact")
plot!(convert_histo(solnet[Int(t+1),:]),linewidth=3,line=:dash,label="SSA")
