using Distributions,Plots
include("../utils.jl")
# SSA bd
# ρ 0 -> N
# d N -> 0

ρ = 30
d = 1

tmax = 30.
S = [1,-1]

function SSA_bd(tmax,saveat)
    n = 0
    t = 0

    t_list = [0.,]
    n_list = [0,]

    while t<tmax
        u1,u2 = rand(Uniform(0,1),2)
        fr = [ρ,d*n]

        # first way to define next reaction
        # λ = sum(fr)
        # τ = -log(u1)/λ
        # r = length(filter(x->x<=0,cumsum(fr).- u2*λ))+1
        
        # second way to define next reaction
        τ1 = rand(Exponential(1/ρ))
        # τ1 = -log(u1)/fr[1]
        τ2 = -log(u2)/fr[2]
        τ,r = findmin([τ1,τ2])

        n = n+S[r]
        t = t+τ
        push!(n_list,n)
        push!(t_list,t)
    end

    n_saveat_list = n_list[searchsortedlast.(Ref(t_list), saveat)]
    return n_saveat_list
end

trajectories = 10000
saveat = 0:0.01:20
n_timepoints = zeros(trajectories,length(saveat))

@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end
    
    n_saveat_list = SSA_bd(tmax,saveat)
    n_timepoints[i,:] = n_saveat_list
end

t = 20
SSA_distriburion = convert_histo(n_timepoints[:,end])
# exact_solution = birth_death_degrade(60,ρ,d,t)
plot!(SSA_distriburion,label="SSA",lw=3)
plot!(0:59,exact_solution,label="exact",lw=3,line=:dash)

i = 100
data = n_timepoints[:,i]
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


using Distributions,StatsPlots,StatsBase,DelimitedFiles

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

struct MyDist <: ContinuousUnivariateDistribution end
L = 200 
function Distributions.rand(d::MyDist)
    # temp 的均值就是τ
    # temp = rand(Uniform(10,40))
    temp = 10
    velo = L/temp
    return velo
end
rand(MyDist())

# SSA tele 
# sigma_on  G* --> G
# sigma_off G --> G*
# ρ         G --> G+N will trigger N => 0 after time τ 

S = [-1  1 0; # G*
      1 -1 0; # G
      0  0 1] # N

function car_event(tmax,saveat,sigma_on,sigma_off,ρ)
    velo_list = []
    location_list = []

    n = [1,0,0]
    t = 0
    t_list = [0.]
    n_list = [n,]
    while t<tmax 
        min_t = 0

        u_list = rand(Uniform(0,1),3)
        fr = [n[1]*sigma_on,n[2]*sigma_off,n[2]*ρ]

        τ_list = -log.(u_list)./fr

        channel = [τ_list;Inf;Inf] 
        # 1.G* --> G 
        # 2.G --> G* 
        # 3.car enter 
        # 4.car leave 
        # 5.car meet

        if length(location_list) != 0
            channel[4] = (L-location_list[1])/velo_list[1]
        end

        if length(location_list)>1
            meet_channel = [(location_list[i]-location_list[i+1])/(velo_list[i+1]-velo_list[i]) for i=1:length(location_list)-1]
            if length(filter(x->x>0,meet_channel))>0
                channel[5] = minimum(filter(x->x>0,meet_channel))
            end
        end

        min_t,event = findmin(channel)

        if length(t_list) != 1
            location_list = location_list + velo_list.*min_t
        end
        
        if event < 3
            n = n .+ S[:,event]
        end

        if event == 3 # a car enter
            vv = rand(MyDist())
            push!(velo_list,vv)
            push!(location_list,0.)
        end

        if event == 4 # a car leave
            filter!(x->x<L,location_list)
            velo_list = velo_list[end-length(location_list)+1:end]
        end

        if event == 5 # car meet
            for i=1:length(location_list)-1
                for j=(i+1):length(location_list)
                    #if location_list[i] == location_list[j]
                    if round(location_list[i], digits=4) == round(location_list[j], digits=4)
                        small_velo = minimum([velo_list[i],velo_list[j]])
                        velo_list[i] = small_velo
                        velo_list[j] = small_velo
                    end
                end
            end
        end

        # n_cars = length(location_list)
        # n[3] = copy(n_cars)

        # n[3] = length(location_list)
        
        n = [n[1],n[2],length(location_list)]
        t=t+min_t

        push!(t_list,t)
        push!(n_list,n)
    end

    n_saveat_list = n_list[searchsortedlast.(Ref(t_list), saveat)]
    n_saveat_list = map(sublist -> sublist[3], n_saveat_list)
    return n_saveat_list
end

sigma_on = 0.3
sigma_off = 0.4
ρ = 2
tmax = 100.

saveat = 0:tmax
trajectories = 10000
n_timepoints = zeros(trajectories,length(saveat))

@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end

    n_saveat_list = car_event(tmax,saveat,sigma_on,sigma_off,ρ)
    n_timepoints[i,:] = n_saveat_list
end

mean_value = [mean(n_timepoints[:,i]) for i=1:length(saveat)]
plot(saveat,mean_value,lw=3,xlabel="t",ylabel="mean",label="SSA_test")

i = 101
SSA_distriburion = convert_histo(n_timepoints[:,i])
plot(SSA_distriburion,label="SSA_test",lw=3)

exp(2.5)

using StatsPlots
μ = 0
σ = sqrt(5)
distribution = LogNormal(μ,σ)
plot(0:50,distribution,lw=3)

μ = 2
σ = sqrt(1)
distribution = LogNormal(μ,σ)
plot!(0:50,distribution,lw=3)



N = 10
μ = 0
σ = 2
distribution = LogNormal(μ,σ)
P = [pdf(distribution,i) for i=0:0.1:N-1]
plot(0:0.1:N-1,P,lw=3,label="(0,2)")

μ = 1
σ = sqrt(2)
distribution = LogNormal(μ,σ)
P = [pdf(distribution,i) for i=0:0.1:N-1]
plot!(0:0.1:N-1,P,lw=3,label="(1,sqrt(2))")




rand(distribution)

mean(distribution)
var(distribution)









