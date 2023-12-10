using Distributions,Plots,DelimitedFiles
using CSV,DataFrames,StatsBase
using Catalyst
using DelaySSAToolkit
using Catalyst.EnsembleAnalysis

include("../../../utils.jl")

train_sol_end_list = []
p_list = []

rn = @reaction_network begin
    ρ1, G --> G + N
    ρ2, G --> G + 2N
    ρ3, G --> G + 3N
    ρ4, G --> G + 4N
    ρ5, G --> G + 5N
    ρ6, G --> G + 6N
    ρ7, G --> G + 7N
    ρ8, G --> G + 8N
    ρ9, G --> G + 9N
    ρ10, G --> G + 10N
end ρ1 ρ2 ρ3 ρ4 ρ5 ρ6 ρ7 ρ8 ρ9 ρ10

burst_sup = 10
jumpsys = convert(JumpSystem, rn, combinatoric_ratelaws = false)

u0 = [1,0]
tf = 300
saveat = 1
de_chan0 = [[]]

p_max_list = [0.4,0.8,1.2]

using Random
for p_max in p_max_list
for seed = 2:3
print([seed,p_max],"\n")
rng = Random.seed!(seed)
p = rand(rng,Uniform(0,p_max),10)
p = sort(p,rev=true)./120
push!(p_list,p)

tspan = (0.0, tf)
aggregatoralgo = DelayRejection()
dprob = DiscreteProblem(jumpsys, u0, tspan, p)

τ = 120.0

delay_trigger_affect! = []
for i in 1:burst_sup
    push!(delay_trigger_affect!, function (integrator, rng)
        append!(integrator.de_chan[1], fill(τ, i))
    end)
end
delay_trigger_affect!
delay_trigger = Dict([Pair(i, delay_trigger_affect![i]) for i in 1:burst_sup])
delay_complete = Dict([Pair(i, [2=>-i]) for i in 1:burst_sup])
delay_interrupt = Dict()

delaysets = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)
djprob = DelayJumpProblem(jumpsys, dprob, aggregatoralgo, delaysets, de_chan0,
                          save_positions = (false, false), save_delay_channel = false)
seed = 3
solution = @time solve(djprob, SSAStepper(), seed = seed, saveat = 1)

Sample_size = Int(100000)
ens_prob = EnsembleProblem(djprob)
ens = @time solve(ens_prob, SSAStepper(), EnsembleThreads(), trajectories = Sample_size,
            saveat = 1)
sol_end = componentwise_vectors_timepoint(ens, tf)[2]

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
end

train_sol_end_list
p_list

plot(train_sol_end_list)

train_solnet = zeros(100,length(p_list))
for i = 1:length(p_list)
    train_solnet[:,i] = train_sol_end_list[i][1:100]
end
train_solnet

p_solnet = zeros(10,length(p_list))
for i = 1:length(p_list)
    p_solnet[:,i] = p_list[i][1:10]
end
p_solnet


# plot!(train_sol_end,title=join(["i=",burst_sup]))


using DataFrames,CSV
df = DataFrame(p_solnet,:auto)
CSV.write("Birth-Death/Control_topology/after20231210/p.csv",df)

using DataFrames,CSV
df = DataFrame(train_solnet,:auto)
CSV.write("Birth-Death/Control_topology/after20231210/train_data_ssa2.csv",df)

# mean
using Catalyst.EnsembleAnalysis
tmax = Int(tf+1)
solnet = zeros(tmax,Sample_size)
for i =1:tmax
    solnet[i,:] = componentwise_vectors_timepoint(ens, Int(i-1))[2]
end
solnet

using StatsBase,Plots
mean_SSA_X = [mean(solnet[i,:]) for i=1:size(solnet,1)]
plot(mean_SSA_X[1:tmax],linewidth = 3,legend=:bottomright,label="G0=1")



