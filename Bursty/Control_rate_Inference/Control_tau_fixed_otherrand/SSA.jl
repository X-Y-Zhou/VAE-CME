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

# mean = 120
# Uniform(0,240)   var = 4800  
# Uniform(30,210)  var = 2700
# Uniform(60,180)  var = 1200
# Uniform(90,150)  var = 300
# Uniform(120,120) var = 0    

E(a,b) = (a+b)/2
D(a,b) = (a-b)^2/12

a = 120;b = 120
Ex = E(a,b)
Dx = D(a,b)
L = 200

# reaction rate
# # set1
# λ = 0.0282
# β = 3.46

# # set2
# λ = 0.0082
# β = 1.46

# set3
λ = 0.0182
β = 2.46

# # set4
# λ = 0.0232
# β = 2.96

struct MyDist <: ContinuousUnivariateDistribution end
function Distributions.rand(d::MyDist)
    # temp = rand(Uniform(a,b))
    temp = 120
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

# i = 150
# t = saveat[i]+1
# car_distribution_machi = convert_histo(n_cars_timepoints[i])
# people_distribution_machi = convert_histo(n_people_timepoints[i])
# plot(people_distribution_machi,linewidth=3,label="car")

solnet_people = zeros(length(saveat),trajectories)
for i=1:length(saveat)
    solnet_people[i,:] = n_people_timepoints[i]
end
solnet_people

maximum(solnet_people)

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

title = [join([a,"-",b])]
df = DataFrame(reshape(train_sol_people[:,end],N+1,1),title)
CSV.write("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/data/set3/$(a)-$(b).csv",df)




plot(0:N,train_sol_people[:,end],lw=3,label="car 0 240")
plot!(0:N,train_sol_1[:,1],lw=3,label="car 120")
plot!(0:N,bursty(N+1,120,0.0282,3.46),lw=3,line=:dash,label="exact-120")

function bursty(N,τ,a,b)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;
bursty(64,120,0.0182,2.46)

train_sol = readdlm("Bursty/Control_rate_Inference/control_tau/data/training_data.csv",',')[2:end,:]
train_sol_2 = readdlm("Bursty/Control_rate_Inference/control_tau/data/30-210.csv",',')[2:end,:]
train_sol_3 = readdlm("Bursty/Control_rate_Inference/control_tau/data/60-180.csv",',')[2:end,:]
train_sol_4 = readdlm("Bursty/Control_rate_Inference/control_tau/data/90-150.csv",',')[2:end,:]
train_sol_5 = readdlm("Bursty/Control_rate_Inference/control_tau/data/120-120.csv",',')[2:end,:]

df = DataFrame(train_sol,:auto)
CSV.write("Bursty/Control_rate_Inference/control_tau/data/training_data.csv",df)


train_sol = train_sol[:,1:5]
plot(0:N,train_sol[:,1],lw=1,label="car 0-240")
plot!(0:N,train_sol[:,2],lw=1,label="car 30-120")
plot!(0:N,train_sol[:,3],lw=1,label="car 60-180")
plot!(0:N,train_sol[:,4],lw=1,label="car 90-150")
plot!(0:N,train_sol[:,5],lw=1,label="car 120-120")
plot!(0:N,bursty(N+1,40,0.0282,3.46),lw=1,line=:dash,label="exact-40")


time_choose = 300
plot(0:N,train_sol_1[:,time_choose+1],linewidth = 3,label="133",title=join(["t=",time_choose]))
plot!(0:N,train_sol_2[:,time_choose+1],linewidth = 3,label="533",title=join(["t=",time_choose]))
plot!(0:N,train_sol_3[:,time_choose+1],linewidth = 3,label="1200",title=join(["t=",time_choose]))
plot!(0:N,train_sol_4[:,time_choose+1],linewidth = 3,label="2133",title=join(["t=",time_choose]))
plot!(0:N,train_sol_5[:,time_choose+1],linewidth = 3,label="3333",title=join(["t=",time_choose]))


# plot!(0:N-1,birth_death(N,16),linewidth=3,label="tau=16")

#=
此段代码仅考虑了车辆碰撞的问题，考虑方法是记录每一辆车的位置以及对应位置上的速度，由于碰撞时两车速度会变成一样，因此在记录车辆离开时间时用的最前面车的位置和速度
channel[2] = (L-location_list[1])/velo_list[1]
当车速不同时，考虑超越的情况，不能用第一辆车的位置和速度来确定车离开的时间，造成问题
=#


using SymPy,QuadGK
@vars x s
L = 200
T1 = 10
T2 = 40
p = 0.6
λ = 0.0282
t = 10

f(x) = x<T1 ? 0 : T1<x<T2 ? (x-10)/30 : 1
f_(x) = 1-f(x)

g0(x) = f_(t-x)
P0 = exp(-λ*quadgk(g0, 0, t, rtol=1e-3)[1])

n = 1
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
