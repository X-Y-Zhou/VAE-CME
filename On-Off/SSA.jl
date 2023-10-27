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
tf = 120.
tspan = (0, tf)
p = [0.0282, 1., 3.46]
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
last_slice = componentwise_vectors_timepoint(ens, 0.0)

Sample_size = 10^5
using Catalyst.EnsembleAnalysis
tmax = 121
solnet = zeros(tmax,Sample_size)
# for i in 0:0.1:10
#     solnet[Int(i*10+1),:] = componentwise_vectors_timepoint(ens, i)[3]
# end

for i in 1:1:121
    solnet[i,:] = componentwise_vectors_timepoint(ens, i-1)[3]
end

solnet

data = solnet
train_sol_SSA = zeros(N+1,size(data,1))
for i =1:size(data,1)
    probability = convert_histo(vec(data[i,:]))[2]
    if length(probability)<N+1
        train_sol_SSA[1:length(probability),i] = probability
    else
        train_sol_SSA[1:N+1,i] = probability[1:N+1]
    end
end
train_sol_SSA

train_sol
N
function plot_distribution(time_choose)
    p=plot(0:N,train_sol_SSA[:,time_choose+1],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,bursty(N+1,0.0282,3.46,time_choose),linewidth = 3,label="SSA",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(5)
    p2 = plot_distribution(10)
    p3 = plot_distribution(15)
    p4 = plot_distribution(20)
    p5 = plot_distribution(30)
    p6 = plot_distribution(40)
    p7 = plot_distribution(50)
    p8 = plot_distribution(60)
    p9 = plot_distribution(70)
    p10 = plot_distribution(80)
    p11 = plot_distribution(90)
    p12 = plot_distribution(120)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()