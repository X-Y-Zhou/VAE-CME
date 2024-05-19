using Distributions,Plots
# bd delay help ziyan 

S = [1.]
ρ = 3
τ = 10
tmax = 20

function SSA_bd(tmax,saveat)
    n = 0
    t = 0

    t_list = [0.,]
    n_list = [0,]

    Pk = zeros(length(S))
    Tk = zeros(length(S))
    sl = [Inf]
    Pk = log.(1.0./rand(Uniform(0,1),length(S)))

    while t<tmax
        fr = [ρ]
        Δtk = (Pk .- Tk)./fr

        minΔtk,r = findmin(Δtk)
        Δ,event = findmin([minΔtk,sl[1]-t])
        t = t+Δ
        print(event,"\n")
        if event == 1
            n = n + S[r]
            pushfirst!(sl,t+τ)
            Pk[r] = Pk[r]+log(1/rand(Uniform(0,1)))
        end

        if event == 2 
            n = n-1
            deleteat!(sl,1)
        end
        Tk = Tk .+ fr.* Δ

        push!(n_list,n)
        push!(t_list,t)
    end

    n_saveat_list = n_list[searchsortedlast.(Ref(t_list), saveat)]
    return n_saveat_list
end

trajectories = 10000
saveat = 0:0.01:20
n_timepoints = zeros(trajectories,length(saveat))

SSA_bd(tmax,saveat);

@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end
    
    n_saveat_list = SSA_bd(tmax,saveat)
    n_timepoints[i,:] = n_saveat_list
end

mean_value = [mean(n_timepoints[:,i]) for i=1:length(saveat)]
plot(saveat,mean_value,lw=3,xlabel="t",ylabel="mean")

t = 20
SSA_distriburion = convert_histo(n_timepoints[:,end])



