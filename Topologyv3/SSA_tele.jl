include("../utils.jl")
include("SSA_car_utils.jl")

# tele
# μ = 0
# σ = sqrt(2)
# dist = LogNormal(μ,σ)
# mean(dist)

dist = Uniform(0,20)
dist = 10
# sigma_on,sigma_off,ρ = ps_list[20]
# sigma_on = 0.003
# sigma_off = 0.004
# ρ = 3

sigma_on,sigma_off,ρ = [0.0037389896785794694,0.0036070583793114874,4.059660383260196]

L = 200
tmax = 100.
N = 80

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