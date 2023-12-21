using Random, Distributions
using DelaySSAToolkit,Catalyst
using Catalyst.EnsembleAnalysis,Plots

include("../../../utils.jl")

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


p_list = [[0.25,1,1.5]]
p_list = [[0.003,0.005,0.3],[0.003,0.008,0.3],[0.006,0.008,0.3],[0.006,0.012,0.3],
          [0.003,0.005,1],[0.003,0.008,1],[0.006,0.008,1],[0.006,0.012,1],
          [0.01,0.015,0.3],[0.01,0.05,0.3],[0.03,0.06,0.3],[0.03,0.1,0.3],
          ]

# p_list = [[0.5,1,0.25]]
train_sol_end_list = []
p_list = [[0.003,0.004,0.3]]
# for p in p_list
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

Sample_size = Int(1e6)
ensprob = EnsembleProblem(djprob)
@time ens = solve(ensprob, SSAStepper(), EnsembleThreads(); trajectories=Sample_size,saveat=1.)
last_slice = componentwise_vectors_timepoint(ens, tf)

Goff = componentwise_vectors_timepoint(ens, tf)[1]
Gon = componentwise_vectors_timepoint(ens, tf)[2]
NRNA = componentwise_vectors_timepoint(ens, tf)[3]

N = 100
P0_list = []
P1_list = []
for i=1:Sample_size
    if Gon[i]==0
        push!(P0_list,NRNA[i])
    else
        push!(P1_list,NRNA[i])
    end
end
P0_list
P1_list

P0_temp = (counts(Int.(P0_list))./Sample_size)
P1_temp = (counts(Int.(P1_list))./Sample_size)
P0 = zeros(N)
P1 = zeros(N)

if length(P0_temp)<N
    P0[1:length(P0_temp)] = P0_temp
else
    P0[1:N] = P0_temp[1:N]
end

if length(P1_temp)<N
    P1[1:length(P1_temp)] = P1_temp
else
    P1[1:N] = P1_temp[1:N]
end

train_sol_end = zeros(N)
probability = convert_histo(vec(NRNA))[2]
# probability = (counts(Int.(NRNA))./Sample_size)
if length(probability)<N
    train_sol_end[1:length(probability)] = probability
else
    train_sol_end[1:N] = probability[1:N]
end
plot(train_sol_end)
plot(P0,label="P0")
plot(P1,label="P1")
push!(train_sol_end_list,train_sol_end)
# end
P0_P1 = [P0;P1]
train_sol_end_list
# plot(train_sol_end_list[end],lw=3,label=p_list[1])
plot(train_sol_end_list,lw=3)

train_sol_end[1:70]
plot(0:N-1,solution,linewidth = 3,label="topo",xlabel = "# of products \n", ylabel = "\n Probability")
plot(0:N-1,train_sol_end_list,lw=3,label="SSA",line=:dash)

train_sol_end = train_sol_end_list[1]
using DataFrames,CSV
df = DataFrame(reshape(P0_P1,2*N,1),:auto)
CSV.write("Birth-Death/Control_topology/after20231221/ssa_tele.csv",df)


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