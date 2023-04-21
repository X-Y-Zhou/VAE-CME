## This example shows an application of `DelaySSA.jl` for a bursty model with delay

## A Bursty model with delay is described as 
# ab^n/(1+b)^(n+1): 0 -> n P, which triggers n P to degrade after delay time τ

using DelaySSAToolkit
using Catalyst
begin # construct reaction network
    @parameters a b t
    @variables X(t)
    burst_sup = 30
    rxs = [Reaction(a * b^i / (1 + b)^(i + 1), nothing, [X], nothing, [i])
           for i in 1:burst_sup]
    rxs = vcat(rxs)
    @named rs = ReactionSystem(rxs, t, [X], [a, b])
end

# convert the ReactionSystem to a JumpSystem
jumpsys = convert(JumpSystem, rs, combinatoric_ratelaws = false)

u0 = [0]
de_chan0 = [[]]
tf = 800.0
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
jprob = DelayJumpProblem(jumpsys, dprob, alg, delayjumpset, de_chan0,
                         save_positions = (true, true), save_delay_channel = false)
# saveat = 0:1:tf
seed = 4
@time sol = solve(jprob, SSAStepper(), seed = seed)

ensprob = EnsembleProblem(jprob)
traj = 100000
@time ens = solve(ensprob, SSAStepper(), EnsembleSerial(), trajectories = traj)


#convert a vector to Probability distributions
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

using Catalyst.EnsembleAnalysis,StatsBase
N = 64
sol_bursty = zeros(N+1,Int(tf)+1)
for i=1:Int(tf)+1
    people_distribution_machi = convert_histo(componentwise_vectors_timepoint(ens, i-1)[1])
    if length(people_distribution_machi[2])>N+1
        sol_bursty[:,i] = people_distribution_machi[2][1:N+1]
    else
        sol_bursty[1:length(people_distribution_machi[2]),i] = people_distribution_machi[2][1:length(people_distribution_machi[2])]
    end
end
sol_bursty

# 100
# 300
# 1000
# 3000
# 10000
# 100000

using DataFrames,CSV
df = DataFrame(sol_bursty,:auto)
CSV.write("machine-learning/ode/bursty/Reduce_sample_size/$traj.csv",df)

using Plots
N = 64
function plot_distribution(time_choose)
    # p=plot(0:N,solution[:,time_choose],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    p=plot(0:N,sol_bursty[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    # plot!(0:N,train_sol_5400[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    # plot!(0:N,train_sol_7350[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    # plot!(0:N,train_sol_9600[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    # plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(27)
    p2 = plot_distribution(47)
    p3 = plot_distribution(67)
    p4 = plot_distribution(77)
    p5 = plot_distribution(87)
    p6 = plot_distribution(97)
    p7 = plot_distribution(107)
    p8 = plot_distribution(120)
    p9 = plot_distribution(200)
    p10 = plot_distribution(300)
    p11 = plot_distribution(500)
    p12 = plot_distribution(800)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()



