using DelaySSAToolkit
using Catalyst
# using Plots
################################################################################
# the following example illustrates how to deal with a oscillatory system
# d: X-> 0; k1/(1+Y^2): 0-> X; [trigger X-> Y after τ time;] k2*Y/(1+Y):  Y-> 0;

rn = @reaction_network begin
    1 / (1 + Y^2), 0 --> X
    1 / (1 + Y), Y --> 0
end

jumpsys = convert(JumpSystem, rn; combinatoric_ratelaws=false)

u0 = [0, 0]
de_chan0 = [[]]
tf = 400.0
tspan = (0, tf)
τ = 20.0
dprob = DiscreteProblem(jumpsys, u0, tspan)
# jumpsys.dep_graph

delay_trigger_affect! = function (integrator, rng)
    append!(integrator.de_chan[1], τ)
end
delay_trigger = Dict(1 => delay_trigger_affect!)
delay_complete = Dict(1 => [2 => 1, 1 => -1])
delay_interrupt = Dict()
delayjumpset = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)
# alg = DelayRejection()
# alg = DelayDirect()
# alg = DelayMNRM()
alg = DelayDirectCR()
djprob = DelayJumpProblem(jumpsys, dprob, alg, delayjumpset, de_chan0)
sol = solve(djprob, SSAStepper(); seed=12345)

Sample_size = Int(1e4)
ens_prob = EnsembleProblem(djprob)
ens = solve(ens_prob, SSAStepper(), EnsembleThreads(); trajectories=Sample_size, saveat=1)

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

N = 50
tmax = Int(tf+1)
sol_N = zeros(N,tmax)
for i =1:tmax
    sol_N[:,i] = embeding_dist(convert_histo(vec(solnet[i,:]))[2],N)
end
writedlm("SSAcode/oscillation_X.txt",sol_N)



using Catalyst.EnsembleAnalysis
tmax = Int(tf+1)
solnet = zeros(tmax,Sample_size)
for i =1:tmax
    solnet[i,:] = componentwise_vectors_timepoint(ens, Int(i-1))[2]
end
solnet

using StatsBase,Plots
mean_SSA_X = [mean(solnet[i,:]) for i=1:size(solnet,1)]
plot(mean_SSA_X[1:tmax],label="X",linewidth = 3,legend=:bottomright)

N = 50
tmax = Int(tf+1)
sol_N = zeros(N,tmax)
for i =1:tmax
    sol_N[:,i] = embeding_dist(convert_histo(vec(solnet[i,:]))[2],N)
end
plot(sol_N[:,end],label="SSA",lw=3)
plot!(birth_death_delay(N,ρ,τ),lw=3,line=:dash)

writedlm("SSAcode/oscillation_Y.txt",sol_N)

