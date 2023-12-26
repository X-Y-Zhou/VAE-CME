using Random, Distributions
using DelaySSAToolkit,Catalyst
using Catalyst.EnsembleAnalysis

rn = @reaction_network begin
    kon, Goff --> Gon
    koff, Gon --> Goff
    ρ, Gon --> Gon + N
end kon koff ρ
jumpsys = convert(JumpSystem, rn; combinatoric_ratelaws=false)

u0 = [0, 1, 0]
de_chan0 = [[]]
tf = 200.
tspan = (0, tf)

p_list = [[0.003,0.004,0.3],[0.003,0.008,0.3],[0.003,0.015,0.3],
          [0.0045,0.006,0.3],[0.0045,0.008,0.3],[0.0045,0.01,0.3],
          [0.006,0.0075,0.3],[0.006,0.01,0.3],[0.006,0.012,0.3],
          [0.007,0.008,0.3],[0.007,0.01,0.3],[0.006,0.015,0.3],
          [0.008,0.009,0.3],[0.008,0.015,0.3],[0.008,0.02,0.3],
          ]
p_list = [[0.5,1,0.25],[0.5,1,0.5],[0.5,1,0.6],[0.5,1,0.8],[0.5,1,1.0],
          [0.75,1,0.2],[0.75,1,0.4],[0.75,1,0.5],[0.75,1,0.6],[0.75,1,0.75],
          [1.0,1,0.2],[1.0,1,0.3],[1.0,1,0.4],[1.0,1,0.5],[1.0,1,0.55]]
# p_list = [[0.5,1,0.25]]
train_sol_end_list_2 = []

for p in p_list
# p = p_list[1]
print(p,"\n")
# p = [0.08, 1., 2.3]
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
    jumpsys, dprob, DelayRejection(), delayjumpset, de_chan0; save_positions=(false, false)
)

ensprob = EnsembleProblem(djprob)
@time ens = solve(ensprob, SSAStepper(), EnsembleThreads(); trajectories=5e4,saveat=1.)
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
push!(train_sol_end_list_2,train_sol_end)
end

train_sol_end_list
# plot(train_sol_end_list[end],lw=3,label=p_list[1])
plot(train_sol_end_list,lw=3)

train_sol_end[1:70]
plot(0:N-1,solution,linewidth = 3,label="topo",xlabel = "# of products \n", ylabel = "\n Probability")
plot(0:N-1,train_sol_end_list,lw=3,label="SSA",line=:dash)

train_sol_end = train_sol_end_list[1]
using DataFrames,CSV
df = DataFrame(reshape(train_sol_end,N,1),:auto)
CSV.write("Birth-Death/Control_topology/after20231212/ssa_tele.csv",df)


function bursty(N,a,b,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

a = 0.005;
b = 35;
τ = 120;

N = 200
train_sol = bursty(N,a,b,τ)
sum(train_sol)

plot(train_sol)