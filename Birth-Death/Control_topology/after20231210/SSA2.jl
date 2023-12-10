using Distributions,Plots,DelimitedFiles
using CSV,DataFrames,StatsBase
using Catalyst
using DelaySSAToolkit
using Catalyst.EnsembleAnalysis

include("../../../utils.jl")

# train_sol_end_list = []

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

using Random
i = 3
rng = Random.seed!(i)
p = rand(rng,Uniform(0,0.4),10)
# p = p./120
p = sort(p,rev=true)./120
sum(p)

# p = [0.088,0.088/7]

# a = 0.0282
# b = 3.46
# p = [a * b^i / (1 + b)^(i + 1) for i=1:burst_sup]

# for p in p_list
# print(p)
tspan = (0.0, tf)
aggregatoralgo = DelayRejection()
# aggregatoralgo = DelayMNRM()
# aggregatoralgo = DelayDirect()
# aggregatoralgo = DelayDirectCR()
dprob = DiscreteProblem(jumpsys, u0, tspan, p)

τ = 120.0

# delay_trigger_affect! = function (integrator, rng)
#     append!(integrator.de_chan[1], fill(τ, 1))
# end

# delay_trigger_affect2! = function (integrator, rng)
#     append!(integrator.de_chan[1], fill(τ, 2))
# end

# # delay_trigger = Dict([Pair(i, delay_trigger_affect!) for i in 1:burst_sup])
# delay_trigger = Dict(1 => delay_trigger_affect!,2 => delay_trigger_affect2!)
# # delay_trigger = Dict(1 => delay_trigger_affect!)

# # delay_complete = Dict([Pair(i, [2 => -i]) for i in 1:burst_sup])
# delay_complete = Dict(1 => [2 => -1], 2=> [2 => -2])
# # delay_complete = Dict(1 => [2 => -2])
# delay_interrupt = Dict()


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
plot(train_sol_end,title=join(["i=",burst_sup]))


using DataFrames,CSV
df = DataFrame(reshape(p,10,1),:auto)
CSV.write("Birth-Death/Control_topology/after20231205-2/p.csv",df)

using DataFrames,CSV
df = DataFrame(reshape(train_sol_end,N+1,1),:auto)
CSV.write("Birth-Death/Control_topology/after20231205-2/train_data_ssa2.csv",df)

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



