include("../utils.jl")
include("SSA_car_utils.jl")

ps_matrix = readdlm("Topologyv3/tele/data/ps_telev1.txt")
batchsize = size(ps_matrix,2)

# tele
dist = Uniform(0,200)

# sigma_on,sigma_off,ρ = ps_list[20]
# sigma_on = 0.003
# sigma_off = 0.004
# ρ = 3

sigma_on,sigma_off,ρ = ps_matrix[:,26]

L = 200
tmax = 300.
N = 120

saveat = 0:tmax
trajectories = 10000
n_timepoints = zeros(trajectories,length(saveat))
# @time car_event_tele(tmax,saveat,sigma_on,sigma_off,ρ,dist,L)

@time for i =1:trajectories
    if i/100 in [j for j=1.:trajectories/100.]
        print(i,"\n")
    end

    n_saveat_list = car_event_tele(tmax,saveat,sigma_on,sigma_off,ρ,dist,L)
    n_timepoints[i,:] = n_saveat_list
end

mean_value = [mean(n_timepoints[:,i]) for i=1:length(saveat)]
plot(saveat,mean_value,lw=3,xlabel="t",ylabel="mean",label="SSA_test")

i = Int(tmax+1)
SSA_distriburion = convert_histo(n_timepoints[:,i])
plot(SSA_distriburion,label="SSA_test",lw=3)

tele_exact = tele_delay(N,sigma_on,sigma_off,ρ,mean(dist))[3]
plot!(0:N-1,tele_exact,label="exact",lw=3,line=:dash)