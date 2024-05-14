include("../utils.jl")
include("SSA_car_utils.jl")

ps_matrix = readdlm("Topologyv3/tele/data/ps_telev1.txt")
batchsize = size(ps_matrix,2)

# tele
dist = Uniform(0,200)


sigma_on,sigma_off,ρ = ps_matrix[:,49]

L = 200
tmax = 300.
N = 120

saveat = 0:1:tmax
trajectories = 1000
n_timepoints = zeros(trajectories,length(saveat))

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




include("../utils.jl")
include("SSA_car_utils.jl")

ps_matrix = readdlm("Topologyv3/tele/data/ps_telev1.txt")
batchsize = size(ps_matrix,2)

# tele

T1T2_list = [[0,200],[50,150]]
dist = Uniform(0,200)
matrix_bursty = zeros(N,batchsize)

# for i = 1:batchsize
i = 3
sigma_on,sigma_off,ρ = ps_matrix[:,i]
L = 200
tmax = 700.
N = 120

saveat = 0:1:tmax
trajectories = 1000
n_timepoints = zeros(trajectories,length(saveat))

@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end

    n_saveat_list = car_event_tele(tmax,saveat,sigma_on,sigma_off,ρ,dist,L)
    n_timepoints[i,:] = n_saveat_list
end

t = Int(tmax+1)
SSA_distriburion = convert_histo(n_timepoints[:,t])[2]

if length(SSA_distriburion)<N
    matrix_bursty[1:length(SSA_distriburion),i] = SSA_distriburion
else
    matrix_bursty[1:N+1,i] = SSA_distriburion[1:N+1]
end

using Distributed,Pkg
addprocs(3)
# rmprocs(5)
nprocs()
workers()

@everywhere include("../utils.jl")
@everywhere include("SSA_car_utils.jl")

@everywhere ps_matrix = readdlm("Topologyv3/tele/data/ps_telev1.txt")
@everywhere batchsize = size(ps_matrix,2)

@everywhere L = 200
@everywhere tmax = 500.
@everywhere N = 120
@everywhere T1 = 0
@everywhere T2 = 200
@everywhere dist = Uniform(T1,T2)
# @everywhere dist = 100

@everywhere function generate_SSA(set)
    sigma_on,sigma_off,ρ = ps_matrix[:,set]
    saveat = 0:1:tmax
    trajectories = 50000
    n_timepoints = zeros(trajectories,length(saveat))

    @time for i =1:trajectories
        if i/1000 in [j for j=1.:trajectories/1000.]
            print(i,"\n")
        end

        n_saveat_list = car_event_tele(tmax,saveat,sigma_on,sigma_off,ρ,dist,L)
        n_timepoints[i,:] = n_saveat_list
    end

    t = Int(tmax+1)
    SSA_distriburion = convert_histo(n_timepoints[:,t])[2]

    if length(SSA_distriburion)<N
        SSA_distriburion = vcat(SSA_distriburion,zeros(N-length(SSA_distriburion)))
    else
        SSA_distriburion = SSA_distriburion[1:N]
    end
    return SSA_distriburion
end
matrix_tele = hcat(pmap(set->generate_SSA(set),1:batchsize)...);

writedlm("Topologyv3/tele/data/matrix_tele_$T1-$T2.csv",matrix_tele)

# P1 = generate_SSA(1)
# P2 = generate_SSA(10)
# P3 = generate_SSA(40)

# plot(matrix_tele[:,1],lw=3)
# plot!(P1,lw=3,line=:dash)

# plot(matrix_tele[:,2],lw=3)
# plot!(P2,lw=3,line=:dash)

# plot(matrix_tele[:,3],lw=3)
# plot!(P3,lw=3,line=:dash)

