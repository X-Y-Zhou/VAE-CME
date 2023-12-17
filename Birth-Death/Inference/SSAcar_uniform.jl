using Distributions,StatsBase,DelimitedFiles,DataFrames,CSV,Plots
include("../../utils.jl")

a = 0;b = 240
a = 30;b = 210
a = 60;b = 180
a = 90;b = 150
a = 120;b = 120

ab_list = [[0,240],[30,210],[60,180],[90,150]]
ab_list = [[0,400],[50,350],[100,300],[150,250]]

train_sol_car_end = []
train_sol_car_end

set = 1
λ = 0.093037

for temp in ab_list
# temp = [0,sqrt(8)]
print(temp,"\n")
a = temp[1]
b = temp[2]
L = 200
struct MyDist <: ContinuousUnivariateDistribution end
function Distributions.rand(d::MyDist)
    temp = rand(Uniform(a,b))
    # temp = exp(4)+70
    velo = L/temp
    return velo
end
rand(MyDist())

function car_event(tmax,saveat,λ)
    velo_list = []
    location_list = []
    t = 0
    t_list = [0.]
    n_list = [0]
    while t<tmax 
        min_t = 0
        Δ = rand(Exponential(1/λ))

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
        
        if event == 1 #a car enter
            # vv = 20
            vv = rand(MyDist())
            push!(velo_list,vv)
            push!(location_list,0.)
        end

        if event == 2 #a car leave
            filter!(x->x<L,location_list)
            velo_list = velo_list[end-length(location_list)+1:end]
        end

        if event == 3 #car meet
            for i=1:length(location_list)-1
                for j=(i+1):length(location_list)
                    if round(location_list[i], digits=4) == round(location_list[j], digits=4)
                        small_velo = minimum([velo_list[i],velo_list[j]])
                        velo_list[i] = small_velo
                        velo_list[j] = small_velo
                    end
                end
            end
        end

        n_cars = length(location_list)

        t=t+min_t
        push!(t_list,t)
        push!(n_list,n_cars)
    end

    for i in saveat
        index = length(filter(x->x<=0,t_list.-i))
        push!(n_cars_list,n_list[index])
    end

    # for j in n_cars_list
    #     if j==0
    #         push!(n_people_list,0)
    #     else
    #         push!(n_people_list,sum([rand(Geometric(1/(1+β))) for i=1:j]))
    #     end
    # end
end

n_cars_list = []
# n_people_list=[]

saveat = 0:600:600
length(saveat)
tmax = maximum(saveat)
n_cars_timepoints = [[] for i=1:length(saveat)]
# n_people_timepoints = [[] for i=1:length(saveat)]

trajectories = 10000
@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end
    
    n_cars_list = []
    # n_people_list=[]
    car_event(tmax,saveat,λ)

    for i =1:length(saveat)
        push!(n_cars_timepoints[i],n_cars_list[i])
        # push!(n_people_timepoints[i],n_people_list[i])
    end
end

solnet_car = zeros(length(saveat),trajectories)
for i=1:length(saveat)
    solnet_car[i,:] = n_cars_timepoints[i]
end

N = 100
train_sol_car = zeros(N+1,size(solnet_car,1))
for i =1:size(solnet_car,1)
    probability = convert_histo(vec(solnet_car[i,:]))[2]
    if length(probability)<N+1
        train_sol_car[1:length(probability),i] = probability
    else
        train_sol_car[1:N+1,i] = probability[1:N+1]
    end
end
push!(train_sol_car_end,train_sol_car[:,end])
end
train_sol_car_end

plot(train_sol_car_end)

plot(train_sol_car_end[1],lw=2,label="λ=0.075,τ~Uniform(0,400)")
plot!(train_sol_car_end[2],lw=2,label="λ=0.075,τ~Uniform(50,350)")
plot!(train_sol_car_end[3],lw=2,label="λ=0.075,τ~Uniform(100,300)")

plot!(train_sol_car_end[5],lw=2,label="λ=0.09,τ~Uniform(0,400)")
plot!(train_sol_car_end[6],lw=2,label="λ=0.09,τ~Uniform(50,350)")
plot!(train_sol_car_end[7],lw=2,label="λ=0.09,τ~Uniform(100,300)",size=(1600,1200))

plot!(train_sol_car_end[5:7])
plot!(0:N-1,birth_death(N,λ,200))

18.6074/200
i = 3
meanvalue = P2mean(train_sol_car_end[i])

P_temp = [pdf(Poisson(meanvalue),i-1) for i=1:N+1]
plot(0:N,train_sol_car_end[i])
plot!(0:N,P_temp)

title = [join([μ,"-","sqrt(",round(σ^2),")"])]
df = DataFrame(reshape(train_sol_car[:,end],N+1,1),title)
CSV.write("Birth-Death/Inference/data/set$set/$(μ)-sqrt($(round(σ^2))).csv",df)
# end

set = 1; λ = 0.1
set = 2; λ = 0.2
set = 3; λ = 0.05
set = 4; λ = 0.075
set = 5; λ = 0.025

train_sol_1 = readdlm("Birth-Death/Inference/data/set$set/3.0-sqrt(2.0).csv",',')[2:end,:]
train_sol_2 = readdlm("Birth-Death/Inference/data/set$set/2.0-sqrt(4.0).csv",',')[2:end,:]
train_sol_3 = readdlm("Birth-Death/Inference/data/set$set/1.0-sqrt(6.0).csv",',')[2:end,:]
train_sol_4 = readdlm("Birth-Death/Inference/data/set$set/0.0-sqrt(8.0).csv",',')[2:end,:]

N = 100
plot(0:N,vec(train_sol_1),lw=2,label="τ~LogNormal(3,sqrt(2))+70",title=λ)
plot!(0:N,vec(train_sol_2),lw=2,label="τ~LogNormal(2,sqrt(4))+70")
plot!(0:N,vec(train_sol_3),lw=2,label="τ~LogNormal(1,sqrt(6))+70")
plot!(0:N,vec(train_sol_4),lw=2,label="τ~LogNormal(0,sqrt(8))+70")
plot!(0:N,birth_death(λ,N+1,70+exp(4)),lw=2,label="τ~exp(4)+70")

plot!(0:N,P_exact,lw=3,line=:dash,label="exact-124")

using TaylorSeries
function bursty(N,τ,a,b)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;
P_exact = bursty(65,70+exp(4),0.0282,3.46)

using StatsPlots

μ = μ_max;σ = sqrt(0) # var = 0
μ = 3.990;σ = sqrt(0.002) # var = 0
μ = 3.5;σ = sqrt(1) # var = 0
μ = 3;σ = sqrt(2) # var = 19045
μ = 2;σ = sqrt(4) # var = 159773
μ = 1;σ = sqrt(6) # var = 1199623
μ = 0.5;σ = sqrt(7) # var = 1199623
μ = 0.1;σ = sqrt(7.8) # var = 1199623
μ = 0;σ = sqrt(2*μ_max) # var = 8883129

rand(LogNormal(μ,σ))

plot!(0:100,LogNormal(μ,σ))