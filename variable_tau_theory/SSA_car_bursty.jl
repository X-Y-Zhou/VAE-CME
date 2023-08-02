using Distributions,StatsPlots,StatsBase,DelimitedFiles
using TaylorSeries
struct MyDist <: ContinuousUnivariateDistribution end

p = 0.6
L = 200
t_minimum = L/3
t_maximum = L

function Distributions.rand(d::MyDist)
    temp = rand(Bernoulli(p))
    if temp == true
        velo = L/t_minimum
    end

    if temp == false
        velo = L/t_maximum
    end
    return velo
end

sample_list = []
for i=1:1e6
  push!(sample_list,rand(MyDist()))
end
sample_list
filter!(x->x==3,sample_list)
sample_list
histogram(sample_list,bins=100,label="sample")

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

# reaction rate
λ = 0.0282
β = 3.46

n_cars_list = []
n_people_list=[]

saveat = 1:5:300
length(saveat)
# saveat = [30,60,120,150,200,230]
tmax = maximum(saveat)
n_cars_timepoints = [[] for i=1:length(saveat)]
n_people_timepoints = [[] for i=1:length(saveat)]

trajectories = 100000
@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end
    
    n_cars_list = []
    n_people_list=[]
    car_event(tmax,saveat,λ)
    #print(n_cars_list,"\n")
    for i =1:length(saveat)
        push!(n_cars_timepoints[i],n_cars_list[i])
        push!(n_people_timepoints[i],n_people_list[i])
    end
end

i = 20
t = saveat[i]
car_distribution_machi = convert_histo(n_cars_timepoints[i])
people_distribution_machi = convert_histo(n_people_timepoints[i])
# plot(car_distribution_machi,linewidth=3,label="car")
# plot(people_distribution_machi,linewidth=3,label="people",line=:dash)

# theory
using SymPy,QuadGK
@vars x s
L = 200
T1 = L/3
T2 = L
p = 0.6
λ = 0.0282
t = saveat[i]

f(x) = x<T1 ? 0 : T1<x<T2 ? p : 1
f_(x) = 1-f(x)

g0(x) = f_(t-x)
P0 = exp(-λ*quadgk(g0, 0, t, rtol=1e-3)[1])

n = 1
g(x) = exp(-λ*(t-x))*(λ*(t-x))^(n-1)*λ*(1-f(t-x))*exp(-λ*quadgk(g0, 0, x, rtol=1e-3)[1])/factorial(n-1)
Pn = quadgk(g, 0, t, rtol=1e-3)[1]

n_cars_max = length(car_distribution_machi[1])
car_distribution_theory = zeros(n_cars_max)
car_distribution_theory[1] = exp(-λ*quadgk(g0, 0, t, rtol=1e-3)[1])
for i = 2:n_cars_max
    n = i-1
    g(x) = exp(-λ*(t-x))*(λ*(t-x))^(n-1)*λ*(1-f(t-x))*exp(-λ*quadgk(g0, 0, x, rtol=1e-3)[1])/factorial(n-1)
    car_distribution_theory[i] = quadgk(g, 0, t, rtol=1e-3)[1]
end
car_distribution_theory

truncation = 70
sum([(1/(1+β))^m*car_distribution_theory[m+1] for m=0:n_cars_max-1])
sum([pdf(NegativeBinomial(k, 1/(1+β)),0)*car_distribution_theory[k+1] for k=1:n_cars_max-1])+car_distribution_theory[1]
people_distribution_theory = [sum([pdf(NegativeBinomial(k, 1/(1+β)),n)*car_distribution_theory[k+1] for k=1:n_cars_max-1]) for n=0:truncation]
people_distribution_theory[1]=sum([pdf(NegativeBinomial(k, 1/(1+β)),0)*car_distribution_theory[k+1] for k=1:n_cars_max-1])+car_distribution_theory[1]
people_distribution_theory

# check
plot(car_distribution_machi,linewidth=3,label="machine",title="car")
plot!(0:n_cars_max-1,car_distribution_theory,linewidth=3,label="theory",line=:dash)

plot(people_distribution_machi,linewidth=3,label="machine",title="people")
plot!(0:truncation,people_distribution_theory,linewidth=3,label="theory",line=:dash)