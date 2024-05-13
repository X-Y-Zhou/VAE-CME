include("../utils.jl")
include("SSA_car_utils.jl")

# bd
ρ = 0.1
dist = Uniform(0,200)
L = 200
tmax = 600.
N = 30

saveat = 0:tmax
trajectories = 10000
n_timepoints = zeros(trajectories,length(saveat))

@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end

    n_saveat_list = car_event_bd(tmax,saveat,ρ,dist,L)
    n_timepoints[i,:] = n_saveat_list
end

mean_value = [mean(n_timepoints[:,i]) for i=1:length(saveat)]
plot(saveat,mean_value,lw=3,xlabel="t",ylabel="mean",label="SSA_test")

t = 600
SSA_distriburion = convert_histo(n_timepoints[:,t+1])
plot(SSA_distriburion,label="SSA Uniform(0,200)",lw=3)

P_exact = car_exact_bd(T1,T2,ρ,t,N)
plot!(0:N-1,P_exact,lw=3,line=:dash,label="exact")

# bursty
α = 0.0979
β = 4.476
dist = Uniform(0,200)
L = 200
tmax = 600.
N = 120

saveat = 0:tmax
trajectories = 50000
n_timepoints = zeros(trajectories,length(saveat))

@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end

    n_saveat_list = car_event_bursty(tmax,saveat,α,β,dist,L)
    n_timepoints[i,:] = n_saveat_list
end

mean_value = [mean(n_timepoints[:,i]) for i=1:length(saveat)]
plot(saveat,mean_value,lw=3,xlabel="t",ylabel="mean",label="SSA_test")

t = 600
SSA_distriburion = convert_histo(n_timepoints[:,t+1])
plot(SSA_distriburion,label="SSA_test",lw=3)

P_exact = car_exact_bursty(T1,T2,α,β,t,30,N)
plot!(0:N-1,P_exact,lw=3,line=:dash)

# plot!(0:N-1,bursty_delay(N,α,β,mean(dist)),lw=3,line=:dash,label="Uniform(100,100)")

# tele
sigma_on = 0.3
sigma_off = 0.4
ρ = 2
dist = 10
L = 200
tmax = 100.
N = 50

saveat = 0:tmax
trajectories = 50000
n_timepoints = zeros(trajectories,length(saveat))

@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end

    n_saveat_list = car_event_tele(tmax,saveat,sigma_on,sigma_off,ρ,dist,L)
    n_timepoints[i,:] = n_saveat_list
end

mean_value = [mean(n_timepoints[:,i]) for i=1:length(saveat)]
plot(saveat,mean_value,lw=3,xlabel="t",ylabel="mean",label="SSA_test")

i = 101
SSA_distriburion = convert_histo(n_timepoints[:,i])
plot(SSA_distriburion,label="SSA_test",lw=3)

tele_exact = tele_delay(N,sigma_on,sigma_off,ρ,dist)[3]
plot!(0:N-1,tele_exact,label="exact",lw=3,line=:dash)