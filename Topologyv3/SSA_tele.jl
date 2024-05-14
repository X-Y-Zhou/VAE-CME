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

T1T2_list = [[0,200],[100,100]]

matrix_tele_list = []
for j = 1:2
    T1 = T1T2_list[j][1]
    T2 = T1T2_list[j][2]
    matrix_tele = readdlm("Topologyv3/tele/data/matrix_tele_$T1-$T2.csv")
    push!(matrix_tele_list,matrix_tele)
end
matrix_tele_list

function plot_distribution(set)
    plot(0:N-1,matrix_tele_list[1][:,set],linewidth = 3,line=:dash,title=round.(ps_matrix[:,set],digits=4))
    plot!(0:N-1,matrix_tele_list[2][:,set],linewidth = 3,line=:dash)
    # plot!(0:N-1,matrix_tele_list[3][:,set],linewidth = 3,line=:dash)
    # plot!(0:N-1,matrix_tele_list[4][:,set],linewidth = 3,line=:dash)
    # plot!(0:N-1,matrix_tele_list[5][:,set],linewidth = 3,line=:dash)
end
plot_distribution(50)

function plot_channel(i)
    p1 = plot_distribution(1+10*(i-1))
    p2 = plot_distribution(2+10*(i-1))
    p3 = plot_distribution(3+10*(i-1))
    p4 = plot_distribution(4+10*(i-1))
    p5 = plot_distribution(5+10*(i-1))
    p6 = plot_distribution(6+10*(i-1))
    p7 = plot_distribution(7+10*(i-1))
    p8 = plot_distribution(8+10*(i-1))
    p9 = plot_distribution(9+10*(i-1))
    p10 = plot_distribution(10+10*(i-1))
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,layouts=(2,5),size=(1500,600))
end
plot_channel(5)

for i = 1:5
    p = plot_channel(i)
    savefig(p,"Topologyv3/tele/data/compare/fig_$i.svg")
end