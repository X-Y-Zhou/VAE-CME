using Distributions,StatsPlots,StatsBase,DelimitedFiles

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

struct MyDist <: ContinuousUnivariateDistribution end
L = 200 
function Distributions.rand(d::MyDist)
    temp = rand(Uniform(10,40))
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
        
        if event == 1 # a car enter
            vv = rand(MyDist())
            push!(velo_list,vv)
            push!(location_list,0.)
        end

        if event == 2 # a car leave
            filter!(x->x<L,location_list)
            velo_list = velo_list[end-length(location_list)+1:end]
        end

        if event == 3 # car meet
            for i=1:length(location_list)-1
                for j=(i+1):length(location_list)
                    #if location_list[i] == location_list[j]
                    if round(location_list[i], digits=4) == round(location_list[j], digits=4)
                        small_velo = minimum([velo_list[i],velo_list[j]])
                        velo_list[i] = small_velo
                        velo_list[j] = small_velo
                    end
                end
            end
        end
        # print(event," ")

        n_cars = length(location_list)

        t=t+min_t
        push!(t_list,t)
        push!(n_list,n_cars)
    end

    for i in saveat
        index = length(filter(x->x<=0,t_list.-i))
        push!(n_cars_list,n_list[index])
    end
end

# reaction rate
λ = 0.0282

# τ = L/vv
# L = 200
# vv = 20

n_cars_list = []
saveat = [i for i=0:10]
tmax = maximum(saveat)
n_cars_timepoints = [[] for i=1:length(saveat)]

trajectories = 10000
@time for i =1:trajectories
    if i/1000 in [j for j=1.:trajectories/1000.]
        print(i,"\n")
    end
    
    n_cars_list = []
    car_event(tmax,saveat,λ)
    #print(n_cars_list,"\n")
    for i =1:length(saveat)
        push!(n_cars_timepoints[i],n_cars_list[i])
    end
end

n_cars_timepoints
t = 10
car_distribution_machi = convert_histo(n_cars_timepoints[t+1])
plot(car_distribution_machi,linewidth=3,label="car")
plot!(car_distribution_machi[1],y,linewidth=3)
y = [ 0.754273684533089
0.21270517903833108
0.02999143024440468
0.00281919444297404
0.00019875320822966975]

using SymPy,QuadGK
@vars x s
L = 200
T1 = 20
T2 = 220
# p = 0.6
λ = 0.0282
t = 150

f(x) = x<T1 ? 0 : T1<x<T2 ? (x-20)/200 : 1
f_(x) = 1-f(x)

g0(x) = f_(t-x)
P0 = exp(-λ*quadgk(g0, 0, t, rtol=1e-3)[1])

n = 2
g(x) = exp(-λ*(t-x))*(λ*(t-x))^(n-1)*λ*(1-f(t-x))*exp(-λ*quadgk(g0, 0, x, rtol=1e-3)[1])/factorial(n-1)
Pn = quadgk(g, 0, t, rtol=1e-3)[1]

y = zeros(5)
y[1] = P0
for i = 2:length(y)
    n = i-1
    g(x) = exp(-λ*(t-x))*(λ*(t-x))^(n-1)*λ*(1-f(t-x))*exp(-λ*quadgk(g0, 0, x, rtol=1e-3)[1])/factorial(n-1)
    y[i] = quadgk(g, 0, t, rtol=1e-3)[1]
end
y