include("../utils.jl")
include("SSA_car_utils.jl")

# bursty
μ = 0
σ = sqrt(4)
dist = LogNormal(μ,σ)
mean(dist)

# using StatsBase
# P_temp = pdf.(dist,0:0.01:20)
# plot(0:0.01:20,P_temp,lw=3)

α = 0.0282
β = 3.46

L = 200
tmax = 500.
N = 60

saveat = 0:0.01:tmax
trajectories = 10000

# n_saveat_list = car_event_bursty(tmax,saveat,α,β,dist,L)
# n_people_list=[]
# @time for j in n_saveat_list
#     if j==0
#         push!(n_people_list,0)
#     else
#         push!(n_people_list,sum([rand(Geometric(1/(1+β))) for i=1:j]))
#     end
# end
# dist = exp(2)

n_timepoints = zeros(trajectories,length(saveat))
@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end

    n_saveat_list = car_event_bursty(tmax,saveat,α,β,dist,L)
    n_timepoints[i,:] = n_saveat_list
end

dist

mean_value = [mean(n_timepoints[:,i]) for i=1:length(saveat)]
plot(saveat,mean_value,lw=3,xlabel="t",ylabel="mean",label="SSA_test")

i = length(saveat)
SSA_distriburion = convert_histo(n_timepoints[:,i])
plot(SSA_distriburion,label="SSA_test",lw=3)

P2mean(SSA_distriburion[2])

plot(birth_death_delay(60,2.82,exp(2)))

plot!(0:N-1,bursty_delay(N,α,β,mean(dist)),lw=3,line=:dash,label="exact")

