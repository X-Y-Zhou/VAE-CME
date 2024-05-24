using Random, Distributions
using DelaySSAToolkit,Catalyst

rn = @reaction_network begin
    kon, Goff --> Gon
    koff, Gon --> Goff
    ρ, Gon --> Gon + N
end kon koff ρ
jumpsys = convert(JumpSystem, rn; combinatoric_ratelaws=false)

u0 = [1, 0, 0]
de_chan0 = [[]]
tf = 200.
tspan = (0, tf)
p = [0.03, 0.04, 10]
dprob = DiscreteProblem(u0, tspan, p)

delay_trigger_affect! = function (integrator, rng)
    τ = 5
    append!(integrator.de_chan[1], τ)
end
delay_trigger = Dict(3 => delay_trigger_affect!)
delay_complete = Dict(1 => [3 => -1])
delay_interrupt = Dict()
delayjumpset = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)

djprob = DelayJumpProblem(
    jumpsys, dprob, DelayMNRM(), delayjumpset, de_chan0; save_positions=(false, false)
)

Sample_size = Int(1e4)
ens_prob = EnsembleProblem(djprob)
@time ens = solve(ens_prob, SSAStepper(), EnsembleThreads(), trajectories = Sample_size,saveat = 1)

using Catalyst.EnsembleAnalysis
tmax = Int(tf+1)
solnet = zeros(tmax,Sample_size)
for i =1:tmax
    solnet[i,:] = componentwise_vectors_timepoint(ens, Int(i-1))[3]
end
solnet

using StatsBase,Plots
mean_SSA_X = [mean(solnet[i,:]) for i=1:size(solnet,1)]
plot(mean_SSA_X[1:tmax],label="X",linewidth = 3,legend=:bottomright)

N = 100
tmax = Int(tf+1)
sol_N = zeros(N,tmax)
for i =1:tmax
    sol_N[:,i] = embeding_dist(convert_histo(vec(solnet[i,:]))[2],N)
end
plot(sol_N[:,end],label="SSA",lw=3)
plot!(birth_death_delay(N,ρ,τ),lw=3,line=:dash)

writedlm("SSAcode/onoff.txt",sol_N)