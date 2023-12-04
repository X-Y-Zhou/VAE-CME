using Distributions,Plots,DelimitedFiles
using CSV,DataFrames,StatsBase
using Catalyst
using DelaySSAToolkit
using Catalyst.EnsembleAnalysis

include("../../utils.jl")

# rn = @reaction_network begin
#     0.282*N^α/(k+N^α)+d, 0 --> N
# end α k d

# rn = @reaction_network begin
#     0.282*N/(k+N^α)+d, 0 --> N
# end α k d

rn = @reaction_network begin
    1/(k+N^α), 0 --> N
end α k

jumpsys = convert(JumpSystem, rn, combinatoric_ratelaws = false)

u0 = [0.]
tf = 300
saveat = 1
de_chan0 = [[]]

α_list = [0.25,0.5,1,2,3,4]
k_list = [1,5,10,50,100]
p_list = [[α_list[i],k_list[j]] for i=1:length(α_list) for j=1:length(k_list)]

train_sol_end_list = []

# p_list = [[0.25,50,0.1]]
p = [0.25,1]

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
# seed = 3
# solution = @time solve(djprob, SSAStepper(), seed = seed, saveat = 1)

Sample_size = Int(10000)
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

# range = 1:4
# p_list[range]
# plot(train_sol_end_list[range])
# plot(train_sol_end_list)

#write data 
train_solnet = zeros(100,length(p_list))
for i = 1:length(p_list)
    train_solnet[:,i] = train_sol_end_list[i][1:100]
end
train_solnet

using DataFrames,CSV
df = DataFrame(train_solnet,:auto)
CSV.write("Birth-Death/Control_topology/train_data5.csv",df)

# read data 

train_sol = readdlm("Birth-Death/Control_topology/train_data4.csv", ',')[2:end,:]
N = 100

train_sol_list = []
for i = 1:30
    push!(train_sol_list,vec(train_sol[:,i]))
end
plot(train_sol_list)

range = 26:30
p_list[range]
plot(train_sol_list[range])
# plot!(P_12,label="Poisson")

set = 11
ave = P2mean(train_sol_list[set])

distribution = Poisson(ave)
P = zeros(N)
for i=1:N
    P[i] = pdf(distribution,i-1)
end

# distribution = NegativeBinomial(0.282*120,0.57)
# P = zeros(N)
# for i=1:N
#     P[i] = pdf(distribution,i-1)
# end

plot(train_sol_list[set],label=join(["a,k=",p_list[set]]))
plot!(P,label="Poisson")



# mean
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

