using Distributions,Plots,DelimitedFiles
using CSV,DataFrames,StatsBase
using Catalyst
using DelaySSAToolkit

rn = @reaction_network begin
    N/(1+N), 0 --> N
    # 0.1,0 --> N
    0.001, N --> 0
    0.001, N+N --> N
    0.001, N+N --> 0
end

# rn = @reaction_network begin
#     ρ, 0 --> N
# end ρ

jumpsys = convert(JumpSystem, rn, combinatoric_ratelaws = false)

u0 = [1.]
tf = 200
saveat = 1
de_chan0 = [[]]

# ρ = 0.0282*3.46
# α = 0.0282
# k = 0
# p = [ρ,α,k]

# p = [ρ]
tspan = (0.0, tf)
# aggregatoralgo = DelayRejection()
# aggregatoralgo = DelayMNRM()
aggregatoralgo = DelayDirect()
# aggregatoralgo = DelayDirectCR()
dprob = DiscreteProblem(jumpsys, u0, tspan)

τ = 120.0
delay_trigger_affect! = function (integrator, rng)
    append!(integrator.de_chan[1], τ)
end
delay_trigger = Dict(1 => delay_trigger_affect!)
delay_complete = Dict(1 => [1 => -1])

delay_affect1! = function (integrator, rng)
    i = rand(rng, 1:length(integrator.de_chan[1]))
    return deleteat!(integrator.de_chan[1], i)
end

delay_affect2! = function (integrator, rng)
    i = rand(rng, 1:length(integrator.de_chan[1]))
    return deleteat!(integrator.de_chan[1], i)
end

delay_affect3! = function (integrator, rng)
    i = rand(rng, 1:length(integrator.de_chan[1]))
    deleteat!(integrator.de_chan[1], i)
    j = rand(rng, 1:length(integrator.de_chan[1]))
    return deleteat!(integrator.de_chan[1], j)
end


delay_interrupt = Dict(2 => delay_affect1!,3 => delay_affect2!,4 => delay_affect3!)
# delay_interrupt = Dict(2 => delay_affect1!,3 => delay_affect2!)

delay_interrupt = Dict()
delaysets = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)
djprob = DelayJumpProblem(jumpsys, dprob, aggregatoralgo, delaysets, de_chan0,
                          save_positions = (false, false), save_delay_channel = false)
seed = 4
solution = @time solve(djprob, SSAStepper(), seed = seed, saveat = 1)

Sample_size = 100
tmax = Int(tf+1)
solnet = zeros(tmax,Sample_size)

for i = 1:Sample_size
    print(i,"\n")
    solution = solve(djprob, SSAStepper(), seed = i, saveat = 1)
    solnet[:,i] = [solution.u[j][1] for j=1:tf+1]
end



Sample_size = Int(200)
ens_prob = EnsembleProblem(djprob)
ens = @time solve(ens_prob, SSAStepper(), EnsembleThreads(), trajectories = Sample_size,
            saveat = 1)

using Catalyst.EnsembleAnalysis
tmax = Int(tf+1)
solnet = zeros(tmax,Sample_size)
for i =1:tmax
    solnet[i,:] = componentwise_vectors_timepoint(ens, Int(i-1))[1]
end
solnet

N = 65
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
plot(0:N,train_sol[:,120])


using StatsBase,Plots
mean_SSA_X = [mean(solnet[i,:]) for i=1:size(solnet,1)]
plot(mean_SSA_X[1:tmax],label="X",linewidth = 3,legend=:bottomright)

t = 8
plot(birth_death(ρ,t,300),linewidth=3,label="exact")
plot!(convert_histo(solnet[Int(t+1),:]),linewidth=3,line=:dash,label="SSA")

temp = [3,4,5,6]
i = 1
j = 2
deleteat!(temp, [i,j])
