using Random, Distributions
using DelaySSAToolkit,Catalyst
using Catalyst.EnsembleAnalysis

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
p = [0.09317816451905206, 1., 2.2085360276790618]
dprob = DiscreteProblem(u0, tspan, p)

delay_trigger_affect! = function (integrator, rng)
    τ = 120
    append!(integrator.de_chan[1], τ)
end
delay_trigger = Dict(3 => delay_trigger_affect!)
delay_complete = Dict(1 => [3 => -1])
delay_interrupt = Dict()
delayjumpset = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)

djprob = DelayJumpProblem(
    jumpsys, dprob, DelayMNRM(), delayjumpset, de_chan0; save_positions=(false, false)
)

ensprob = EnsembleProblem(djprob)
@time ens = solve(ensprob, SSAStepper(), EnsembleThreads(); trajectories=10^5,saveat=1.)
last_slice = componentwise_vectors_timepoint(ens, tf)

sol_end = componentwise_vectors_timepoint(ens, tf)[3]

N = 100
train_sol_end = zeros(N+1)

probability = convert_histo(vec(sol_end))[2]
if length(probability)<N+1
    train_sol_end[1:length(probability)] = probability
else
    train_sol_end[1:N+1] = probability[1:N+1]
end

train_sol_end[1:70]
solution
plot(solution,linewidth = 3,label="topo",xlabel = "# of products \n", ylabel = "\n Probability")
plot!(0:69,train_sol_end[1:70],lw=3,label="SSA",line=:dash)

plot(0:69,train_sol_end[1:70],lw=3)
plot!(0:69,check_sol,linewidth = 3,label="exact",line=:dash)

using DataFrames,CSV
df = DataFrame(reshape(train_sol_end,N+1,1),:auto)
CSV.write("Birth-Death/Control_topology/after20231205-2/ssa_tele.csv",df)

