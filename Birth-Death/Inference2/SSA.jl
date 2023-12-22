using Distributions,Plots,DelimitedFiles
using CSV,DataFrames,StatsBase
using Catalyst
using DelaySSAToolkit
using Distributions

rn = @reaction_network begin
    ρ, 0 --> N
end ρ

jumpsys = convert(JumpSystem, rn, combinatoric_ratelaws = false)

u0 = [0.]
tf = 200
saveat = 1
de_chan0 = [[]]
ρ = 2.0
p = [ρ]
tspan = (0.0, tf)
# aggregatoralgo = DelayRejection()
# aggregatoralgo = DelayMNRM()
aggregatoralgo = DelayDirect()
# aggregatoralgo = DelayDirectCR()
dprob = DiscreteProblem(u0, tspan, p)

μ_σ_list = [[4,sqrt(0)],[3,sqrt(2)],[2,sqrt(4)],[1,sqrt(6)],[0,sqrt(8)]]
train_sol_end_list = []

for temp in μ_σ_list
print(temp,"\n")
μ = temp[1]
σ = temp[2]
delay_trigger_affect! = function (integrator, rng)
    τ = rand(LogNormal(μ,σ))
    # τ = exp(4)+70
    append!(integrator.de_chan[1], τ)
end
delay_trigger = Dict(1 => delay_trigger_affect!)
delay_complete = Dict(1 => [1 => -1])
delay_interrupt = Dict()
delaysets = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)
djprob = DelayJumpProblem(jumpsys, dprob, aggregatoralgo, delaysets, de_chan0,
                          save_positions = (false, false), save_delay_channel = false)
# seed = 3
# sol = @time solve(djprob, SSAStepper(), seed = seed, saveat = 1)

Sample_size = Int(5e4)
ens_prob = EnsembleProblem(djprob)
@time ens = solve(ens_prob, SSAStepper(), EnsembleThreads(), trajectories = Sample_size,
            saveat = 1)

using Catalyst.EnsembleAnalysis
tmax = Int(tf+1)
solnet = zeros(tmax,Sample_size)
sol_end = componentwise_vectors_timepoint(ens, tf)[1]

N = 200
train_sol_end = zeros(N)

probability = convert_histo(vec(sol_end))[2]
if length(probability)<N
    train_sol_end[1:length(probability)] = probability
else
    train_sol_end[1:N] = probability[1:N]
end
push!(train_sol_end_list,train_sol_end)
end

train_sol_end_matrix = zeros(N,length(μ_σ_list))
for i = 1:length(μ_σ_list)
    train_sol_end_matrix[:,i] = train_sol_end_list[i]
end

train_sol_end_matrix
df = DataFrame(train_sol_end_matrix,:auto)
CSV.write("Birth-Death/Inference2/data/ρ=$ρ.csv",df)


plot(train_sol_end_list)

plot!(birth_death(ρ,84,N))

plot!(birth_death(N,ρ,exp(4)+70))

function birth_death(ρ,t,N)
    distribution = Poisson(ρ*t)
    P = zeros(N)
    for i=1:N
        P[i] = pdf(distribution,i-1)
    end
    return P
end;


using StatsBase,Plots
mean_SSA_X = [mean(solnet[i,:]) for i=1:size(solnet,1)]
plot(mean_SSA_X[1:tmax],label="X",linewidth = 3,legend=:bottomright)


