using Distributions,Plots,DelimitedFiles
using CSV,DataFrames,StatsBase
using Catalyst
using DelaySSAToolkit

include("../../utils.jl")

rn = @reaction_network begin
    0.282*N^α/(k+N^α)+d, 0 --> N
end α k d

jumpsys = convert(JumpSystem, rn, combinatoric_ratelaws = false)

u0 = [0.]
tf = 400
saveat = 1
de_chan0 = [[]]

α_list = [0.25,0.5,1,2,3,4]
k_list = [1,3,5,7,9,11]
d = 0.1
p_list = [[α_list[i],k_list[j],d] for i=1:length(α_list) for j=1:length(k_list)]

# p_list = [[2,5,d]] # α k d
train_sol_end_list = []

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

plot(train_sol_end_list)

train_solnet = zeros(80,length(p_list))
for i = 1:length(p_list)
    train_solnet[:,i] = train_sol_end_list[i][1:80]
end
train_solnet

using DataFrames,CSV
df = DataFrame(train_solnet,:auto)
CSV.write("Birth-Death/Control_topology/train_data.csv",df)


