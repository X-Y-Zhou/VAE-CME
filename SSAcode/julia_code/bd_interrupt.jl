using Distributions,Plots,DelimitedFiles
using CSV,DataFrames,StatsBase
using Catalyst
using DelaySSAToolkit
include("../../utils.jl")

rn = @reaction_network begin
    ρ, 0 --> N
    d, N --> 0 
end ρ d

jumpsys = convert(JumpSystem, rn, combinatoric_ratelaws = false)

u0 = [0.]
tf = 50
saveat = 1
de_chan0 = [[]]
ρ = 10
d = 1
p = [ρ,d]
tspan = (0.0, tf)
# aggregatoralgo = DelayRejection()
# aggregatoralgo = DelayMNRM()
aggregatoralgo = DelayDirect()
# aggregatoralgo = DelayDirectCR()
dprob = DiscreteProblem(u0, tspan, p)

τ = 5
delay_trigger_affect! = function (integrator, rng)
    append!(integrator.de_chan[1], τ)
end
delay_trigger = Dict(1 => delay_trigger_affect!)
delay_complete = Dict(1 => [1 => -1])

delay_affect! = function (integrator, rng)
    i = rand(rng, 1:length(integrator.de_chan[1]))
    return deleteat!(integrator.de_chan[1], i)
end

delay_interrupt = Dict(2 => delay_affect!)
delaysets = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)
djprob = DelayJumpProblem(jumpsys, dprob, aggregatoralgo, delaysets, de_chan0,
                          save_positions = (false, false), save_delay_channel = false)

Sample_size = Int(5e4)
ens_prob = EnsembleProblem(djprob)
@time ens = solve(ens_prob, SSAStepper(), EnsembleThreads(), trajectories = Sample_size,saveat = 1)

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

N = 30
tmax = Int(tf+1)
sol_N = zeros(N,tmax)
for i =1:tmax
    sol_N[:,i] = embeding_dist(convert_histo(vec(solnet[i,:]))[2],N)
end
plot(sol_N[:,end],label="SSA",lw=3)

writedlm("SSAcode/bd_interrupt.txt",sol_N)

matrix = readdlm("SSAcode/oscillation_X.txt")
mean_value = [P2mean(matrix[:,i]) for i = 1:size(matrix,2)]
plot(0:1:size(matrix,2)-1,mean_value,lw=3,label="X")

matrix = readdlm("SSAcode/oscillation_Y.txt")
mean_value = [P2mean(matrix[:,i]) for i = 1:size(matrix,2)]
plot!(0:1:size(matrix,2)-1,mean_value,lw=3,label="Y")

savefig("SSAcode/figs/oscillation.svg")

