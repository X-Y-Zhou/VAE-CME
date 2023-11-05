using Distributions,StatsBase,DelimitedFiles,DataFrames,CSV

function convert_histo(data::Vector)
    # Define histogram edge set (integers)
    max_np = ceil(maximum(data))+1
    min_np = 0
    edge = collect(min_np:1:max_np)
    H = fit(Histogram,data,edge)
    saved=zeros(length(H.weights),2);
    saved[:,1] = edge[1:end-1];
    # Normalize histogram to probability (since bins are defined on integers)
    saved[:,2] = H.weights/length(data);
    return saved[:,1], saved[:,2]
end

E(μ,σ) = exp(μ+σ^2/2)
D(μ,σ) = (exp(σ^2)-1)*exp(2*μ+σ^2)

μ_max = 4
mean = exp(4)+70

μ = μ_max;σ = sqrt(0) # var = 0

μ = 3;σ = sqrt(2) # var = 19045
μ = 2;σ = sqrt(4) # var = 159773
μ = 1;σ = sqrt(6) # var = 1199623
μ = 0;σ = sqrt(2*μ_max) # var = 8883129

μ = 0.5;σ = sqrt(7) 
μ = 3.5;σ = sqrt(1)
μ = 3.7;σ = sqrt(0.6)

E(μ,σ) 
D(μ,σ)

# mean = 70+e^4
# LogNormal(3,sqrt(2))+70 [0]
# LogNormal(2,sqrt(4))+70
# LogNormal(1,sqrt(6))+70
# LogNormal(0,sqrt(8))+70 [1]

# reaction rate
set = 1;λ = 0.0282;β = 3.46
set = 2;λ = 0.0082;β = 1.46
set = 3;λ = 0.0182;β = 2.46
set = 4;λ = 0.0232;β = 2.96
set = 5;λ = 0.0182;β = 2.96
set = 6;λ = 0.0082;β = 3.46
set = 7;λ = 0.0282;β = 1.46
set = 8;λ = 0.0082;β = 2.46
set = 9;λ = 0.0282;β = 2.46
set = 10;λ = 0.0232;β = 2.46
set = 11;λ = 0.0282;β = 2.96
set = 12;λ = 0.0282;β = 1.96
set = 13;λ = 0.0232;β = 3.46
set = 14;λ = 0.0232;β = 1.46


μ_σ_list = [[3,sqrt(2)],[2,sqrt(4)],[1,sqrt(6)],[0,sqrt(8)]]
μ_σ_list = [[3,sqrt(2)],[0,sqrt(8)]]
μ_σ_list = [[2,sqrt(4)],[1,sqrt(6)]]

for temp in μ_σ_list
print(temp,"\n")
μ = temp[1]
σ = temp[2]
L = 200
struct MyDist <: ContinuousUnivariateDistribution end
function Distributions.rand(d::MyDist)
    temp = rand(LogNormal(μ,σ))+70
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

    for j in n_cars_list
        if j==0
            push!(n_people_list,0)
        else
            push!(n_people_list,sum([rand(Geometric(1/(1+β))) for i=1:j]))
        end
    end
end

n_cars_list = []
n_people_list=[]

saveat = 0:60:600
length(saveat)
tmax = maximum(saveat)
n_cars_timepoints = [[] for i=1:length(saveat)]
n_people_timepoints = [[] for i=1:length(saveat)]

trajectories = 500000
@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end
    
    n_cars_list = []
    n_people_list=[]
    car_event(tmax,saveat,λ)

    for i =1:length(saveat)
        push!(n_cars_timepoints[i],n_cars_list[i])
        push!(n_people_timepoints[i],n_people_list[i])
    end
end

solnet_people = zeros(length(saveat),trajectories)
for i=1:length(saveat)
    solnet_people[i,:] = n_people_timepoints[i]
end

N = 64
train_sol_people = zeros(N+1,size(solnet_people,1))
for i =1:size(solnet_people,1)
    probability = convert_histo(vec(solnet_people[i,:]))[2]
    if length(probability)<N+1
        train_sol_people[1:length(probability),i] = probability
    else
        train_sol_people[1:N+1,i] = probability[1:N+1]
    end
end
train_sol_people

title = [join([μ,"-","sqrt(",round(σ^2),")"])]
df = DataFrame(reshape(train_sol_people[:,end],N+1,1),title)
CSV.write("Control_rate_Inference/Control_tau_fixed_otherrand_logn/data/set$set/$(μ)-sqrt($(round(σ^2))).csv",df)
end

set = 1
train_sol_1 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_logn/data/set$set/3.0-sqrt(2.0).csv",',')[2:end,:]
train_sol_2 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_logn/data/set$set/2.0-sqrt(4.0).csv",',')[2:end,:]
train_sol_3 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_logn/data/set$set/1.0-sqrt(6.0).csv",',')[2:end,:]
train_sol_4 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand_logn/data/set$set/0.0-sqrt(8.0).csv",',')[2:end,:]

N = 64
plot(0:N,vec(train_sol_1),lw=2,label="τ~LogNormal(3,sqrt(2))+70")
plot!(0:N,vec(train_sol_2),lw=2,label="τ~LogNormal(2,sqrt(4))+70")
plot!(0:N,vec(train_sol_3),lw=2,label="τ~LogNormal(1,sqrt(6))+70")
plot!(0:N,vec(train_sol_4),lw=2,label="τ~LogNormal(0,sqrt(8))+70")


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