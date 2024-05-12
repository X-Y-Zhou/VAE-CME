using Random, Distributions
using DelaySSAToolkit,Catalyst
using Catalyst.EnsembleAnalysis
include("../utils.jl")

rn = @reaction_network begin
    kon, Goff --> Gon
    koff, Gon --> Goff
    ρ, Gon --> Gon + N
end kon koff ρ
jumpsys = convert(JumpSystem, rn; combinatoric_ratelaws=false)

u0 = [1, 0, 0]
de_chan0 = [[]]
tf = 100.
tspan = (0, tf)

# p_list = [[4,5,5]]

# for p in p_list
p = [0.3,0.4,2]
print(p,"\n")
# p = [0.08, 1., 2.3]
dprob = DiscreteProblem(u0, tspan, p)

delay_trigger_affect! = function (integrator, rng)
    τ = 10
    append!(integrator.de_chan[1], τ)
end
delay_trigger = Dict(3 => delay_trigger_affect!)
delay_complete = Dict(1 => [3 => -1])
delay_interrupt = Dict()
delayjumpset = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)

djprob = DelayJumpProblem(
    jumpsys, dprob, DelayCoevolve(), delayjumpset, de_chan0; save_positions=(false, false)
)

ensprob = EnsembleProblem(djprob)
Sample_size = 1e4
@time ens = solve(ensprob, SSAStepper(), EnsembleThreads(); trajectories=Sample_size,saveat=1.)
last_slice = componentwise_vectors_timepoint(ens, tf)

sol_end = componentwise_vectors_timepoint(ens, tf)[3]

N = 100
train_sol_end = zeros(N)

probability = convert_histo(vec(sol_end))[2]
if length(probability)<N
    train_sol_end[1:length(probability)] = probability
else
    train_sol_end[1:N] = probability[1:N]
end
# push!(train_sol_end_list,train_sol_end)
# end

# train_sol_end_list
# plot(train_sol_end_list[end],lw=3,label=p_list[1])
plot!(0:N-1,train_sol_end,lw=3,line=:dash)


using Catalyst.EnsembleAnalysis
tmax = Int(tf+1)
solnet = zeros(tmax,Int(Sample_size))
for i =1:tmax
    solnet[i,:] = componentwise_vectors_timepoint(ens, Int(i-1))[3]
end

using StatsBase,Plots
mean_SSA = [mean(solnet[i,:]) for i=1:size(solnet,1)]
plot!(0:tmax-1,mean_SSA[1:tmax],label="SSA_tool",linewidth = 3,legend=:bottomright,line=:dash)










