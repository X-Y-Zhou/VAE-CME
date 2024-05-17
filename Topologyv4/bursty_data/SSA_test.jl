include("../../utils.jl")
include("../../SSA_car_utils.jl")

# bd
ρ = 0.1
# dist = Uniform(0,200)
# dist = Uniform(50,150)
dist = LogNormal(1,sqrt(2))
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

t = 1000
SSA_distriburion = convert_histo(n_timepoints[:,t+1])
plot(SSA_distriburion,label="SSA Uniform(0,200)",lw=3)

P_exact = car_exact_bd(dist,ρ,t,N)
plot!(0:N-1,P_exact,lw=3,line=:dash,label="exact")

# bursty
ps_matrix_bursty = readdlm("Topologyv4/ps_burstyv1.csv")

set = 10
α,β = ps_matrix_bursty[:,set]
matrix_bursty = readdlm("Topologyv4/bursty_data/matrix_bursty_μ=0.0.csv")
# α = 0.8
# β = 3
μ = 0
σ = sqrt(4)
dist = LogNormal(μ,σ)+100
# dist = Uniform(5,15)
# dist = 10
L = 200
tmax = 500.
N = 120


saveat = 0:tmax
trajectories = 10000
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

t = 500
SSA_distriburion = convert_histo(n_timepoints[:,t+1])
plot(SSA_distriburion,label="SSA_test",lw=3)
plot!(0:N-1,matrix_bursty[:,set],lw=3,line=:dash)
# plot!(0:N-1,car_exact_bursty(dist,α,β,t,30,N),lw=3,line=:dash)


car_exact_bursty(dist,α,β,t,30,N)

train_sol = readdlm("Topologyv6/bursty_data/matrix_bursty_0-20.csv") # var = max
plot!(0:N-1,train_sol[:,set],label="exact",lw=3)

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