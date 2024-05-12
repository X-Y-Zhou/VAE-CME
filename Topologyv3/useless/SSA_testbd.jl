using Distributions,Plots
include("../utils.jl")

ρ = 5
τ = 10
L = 200 

struct MyDist <: ContinuousUnivariateDistribution end
function Distributions.rand(d::MyDist)
    # temp = rand(Uniform(10,40))
    temp = τ
    velo = L/temp
    return velo
end
rand(MyDist())

function generate_velo(distribution::Distribution,L)
    τ = rand(distribution)
    velo = L/τ
    return velo
end
temp = [generate_velo(Uniform(1,3),L) for i=1:10000]


function car_event(tmax,saveat,ρ)
    velo_list = []
    location_list = []
    t = 0
    t_list = [0.]
    n_list = [0]
    while t<tmax 
        min_t = 0

        u = rand(Uniform(0,1))
        fr = ρ
        Δ = -log(u)/fr

        channel = [Δ,Inf,Inf] #1.car enter 2.car leave 3.car meet

        if length(location_list) != 0
            channel[2] = (L-location_list[1])/velo_list[1]
        end

        if length(location_list)>1
            meet_channel = [(location_list[i]-location_list[i+1])/(velo_list[i+1]-velo_list[i]) for i=1:length(location_list)-1]
            if length(filter(x->x>0,meet_channel))>0
                channel[3] = minimum(filter(x->x>0,meet_channel))
            end
        end

        min_t,event = findmin(channel)

        if length(t_list) != 1
            location_list = location_list + velo_list.*min_t
        end
        
        if event == 1 # a car enter
            vv = rand(MyDist())
            push!(velo_list,vv)
            push!(location_list,0.)
        end

        if event == 2 # a car leave
            filter!(x->x<L,location_list)
            velo_list = velo_list[end-length(location_list)+1:end]
        end

        if event == 3 # car meet
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
        # print(event," ")

        n_cars = length(location_list)

        t=t+min_t
        push!(t_list,t)
        push!(n_list,n_cars)
    end

    n_saveat_list = n_list[searchsortedlast.(Ref(t_list), saveat)]
    return n_saveat_list
end

tmax = 20
trajectories = 1000
saveat = 0:0.01:tmax
n_timepoints = zeros(trajectories,length(saveat))

@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end
    
    n_saveat_list = car_event(tmax,saveat,ρ)
    n_timepoints[i,:] = n_saveat_list
end

mean_value = [mean(n_timepoints[:,i]) for i=1:length(saveat)]
plot(saveat,mean_value,lw=3,xlabel="t",ylabel="mean",label="SSA_test")


t = 20
SSA_distriburion = convert_histo(n_timepoints[:,end])
# exact_solution = birth_death_degrade(60,ρ,d,t)
plot!(SSA_distriburion,label="SSA",lw=3)
plot!(0:59,exact_solution,label="exact",lw=3,line=:dash)

mean_exact = [[P2mean(birth_death(100,ρ,i)) for i=0:τ];[P2mean(birth_death(100,ρ,τ)) for i=1+τ:tmax]]
plot!(0:tmax,mean_exact,lw=3,label="exact",line=:dash)



