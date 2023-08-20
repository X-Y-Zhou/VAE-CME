## This example shows an application of `DelaySSA.jl` for a bursty model with delay

## A Bursty model with delay is described as
# ab^n/(1+b)^(n+1): 0 -> n P, which triggers n P to degrade after delay time τ

using DelaySSAToolkit
using Catalyst
begin # construct reaction network
    @parameters a b t
    @variables X(t)
    burst_sup = 30
    rxs = [
        Reaction(a * b^i / (1 + b)^(i + 1), nothing, [X], nothing, [i]) for i in 1:burst_sup
    ]
    rxs = vcat(rxs)
    @named rs = ReactionSystem(rxs, t, [X], [a, b])
end

# convert the ReactionSystem to a JumpSystem
jumpsys = convert(JumpSystem, rs; combinatoric_ratelaws=false)

u0 = [0]
de_chan0 = [[]]
tf = 600.0
tspan = (0, tf)
ps = [0.0282, 3.46]
dprob = DiscreteProblem(jumpsys, u0, tspan, ps)
τ = 120.0

delay_trigger_affect! = []
for i in 1:burst_sup
    push!(delay_trigger_affect!, function (integrator, rng)
        append!(integrator.de_chan[1], fill(τ, i))
    end)
end
delay_trigger_affect!
delay_trigger = Dict([Pair(i, delay_trigger_affect![i]) for i in 1:burst_sup])
delay_complete = Dict(1 => [1 => -1])
delay_interrupt = Dict()
# alg = DelayRejection()
alg = DelayCoevolve()

delayjumpset = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)
jprob = DelayJumpProblem(
    jumpsys,
    dprob,
    alg,
    delayjumpset,
    de_chan0;
    save_positions=(true, true),
    save_delay_channel=false,
)
# saveat = 0:1:tf
seed = 4
@time sol = solve(jprob, SSAStepper(); seed=seed)


Sample_size = 10000
ensprob = EnsembleProblem(jprob);
@time ens = solve(ensprob, SSAStepper(), EnsembleSerial();trajectories=Sample_size)

using Catalyst.EnsembleAnalysis
solnet = zeros(Int(tf)+1,Sample_size)
for i =1:Int(tf)+1
    solnet[i,:] = componentwise_vectors_timepoint(ens, i-1)[1]
end
solnet

using DataFrames,CSV
df = DataFrame(solnet,:auto)
CSV.write("Bursty/Infrence/data/SSA_3",df)

using Distributions,StatsBase
function convert_histo(data::Vector)
    # Define histogram edge set (integers)
    max_np = ceil(maximum(data))+1
    min_np = 0
    edge = collect(min_np:1:max_np)
    H = fit(Histogram,data,edge)
    saved=zeros(length(H.weights),2);
    saved[:,1] = edge[1:end-1];
    # Normalize histogram to probability (since bins are defined on integers)
    saved[:,2] = H.weights/length(data);
    return saved[:,1], saved[:,2]
end


using TaylorSeries
function bursty(N,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

a = ps[1]
b = ps[2]
N = 64
end_time = Int(tf)
exact_sol = zeros(N+1,end_time+1)
for i = 0:end_time
    if i < τ
        exact_sol[1:N+1,i+1] = bursty(N+1,i)
    else
        exact_sol[1:N+1,i+1] = bursty(N+1,τ)
    end
end

using Plots
t = 120
plot(convert_histo(vec(solnet[t+1,:])),lw=3,label="SSA")
plot!(0:N,exact_sol[:,t+1],lw=3,line=:dash,label="exact")