using Distributions,Plots,DelimitedFiles
using CSV,DataFrames,StatsBase
using Catalyst
using DelaySSAToolkit

include("../../utils.jl")

train_sol_end_list = []

rn = @reaction_network begin
    0.282*N^α/(k+N^α), 0 --> N
end α k

jumpsys = convert(JumpSystem, rn, combinatoric_ratelaws = false)

u0 = [1.]
tf = 800
saveat = 1
de_chan0 = [[]]
p_list = [[2,5],[1,5]] # α k

for p in p_list
print(p)
tspan = (0.0, tf)
aggregatoralgo = DelayRejection()
# aggregatoralgo = DelayMNRM()
# aggregatoralgo = DelayDirect()
# aggregatoralgo = DelayDirectCR()
dprob = DiscreteProblem(jumpsys, u0, tspan, p)

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
# seed = 2
# solution = @time solve(djprob, SSAStepper(), seed = seed, saveat = 1)

Sample_size = Int(50000)
ens_prob = EnsembleProblem(djprob)
ens = @time solve(ens_prob, SSAStepper(), EnsembleThreads(), trajectories = Sample_size,
            saveat = 1)
sol_end = componentwise_vectors_timepoint(ens, tf)[1]

N = 100
train_sol_end = zeros(N+1)

probability = convert_histo(vec(sol_end))[2]
if length(probability)<N+1
    train_sol_end[1:length(probability)] = probability
else
    train_sol_end[1:N+1] = probability[1:N+1]
end
push!(train_sol_end_list,train_sol_end)
end
train_sol_end_list[1]
plot(train_sol_end_list[1])
plot!(train_sol_end_list[2])

train_solnet = zeros(N+1,length(p_list))
for i = 1:length(p_list)
    train_solnet[:,i] = train_sol_end_list[i]
end
train_solnet

using DataFrames,CSV
df = DataFrame(train_solnet,:auto)
CSV.write("Birth-Death/Control_topology/train_data.csv",df)


using Catalyst.EnsembleAnalysis
tmax = Int(tf+1)
solnet = zeros(tmax,Sample_size)
for i =1:tmax
    solnet[i,:] = componentwise_vectors_timepoint(ens, Int(i-1))[1]
end
solnet

N = 200
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
# plot(0:N,train_sol[:,200])

# using StatsPlots
# P_0_distribution = NegativeBinomial(0.5*120, 1/3.46);
# plot(0:N,train_sol[:,200])
# plot(P_0_distribution)

using StatsBase,Plots
mean_SSA_X = [mean(solnet[i,:]) for i=1:size(solnet,1)]
plot!(mean_SSA_X[1:tmax],label="X",linewidth = 3,legend=:bottomright)

