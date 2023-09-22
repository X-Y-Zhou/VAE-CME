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

# mean = 180
# Uniform(0,360)    var = 10800  
# Uniform(45,315)   var = 6075
# Uniform(90,270)   var = 2700
# Uniform(135,225)  var = 675
# Uniform(180,180)  var = 0    

E(a,b) = (a+b)/2
D(a,b) = (a-b)^2/12

a = 0;b = 360
a = 45;b = 315
a = 90;b = 270
a = 135;b = 225
a = 180;b = 180

Ex = E(a,b)
Dx = D(a,b)
L = 200

# reaction rate
set = 1
λ = 0.0282
β = 3.46

# set = 2
# λ = 0.0082
# β = 1.46

# set = 3
# λ = 0.0182
# β = 2.46

# set = 4
# λ = 0.0232
# β = 2.96

struct MyDist <: ContinuousUnivariateDistribution end
function Distributions.rand(d::MyDist)
    temp = rand(Uniform(a,b))
    # temp = 180
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
CSV.write("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand2/data/set$set/$(a)-$(b).csv",df)

bursty(N,0.0282,3.46,180)



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

train_sol_1 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand2/data/set1/0-360.csv",',')[2:end,:]
train_sol_2 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand2/data/set1/45-315.csv",',')[2:end,:]
train_sol_3 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand2/data/set1/90-270.csv",',')[2:end,:]
train_sol_4 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand2/data/set1/135-225.csv",',')[2:end,:]
train_sol_5 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand2/data/set1/180-180.csv",',')[2:end,:]

plot(0:N,vec(train_sol_1),lw=3,label="car 1")
plot!(0:N,vec(train_sol_2),lw=3,label="car 2")
plot!(0:N,vec(train_sol_3),lw=3,label="car 3")
plot!(0:N,vec(train_sol_4),lw=3,label="car 4")
plot!(0:N,vec(train_sol_5),lw=3,label="car 5")

train_sol_1 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/data/set1/0-240.csv",',')[2:end,:]
train_sol_2 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/data/set1/30-210.csv",',')[2:end,:]
train_sol_3 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/data/set1/60-180.csv",',')[2:end,:]
train_sol_4 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/data/set1/90-150.csv",',')[2:end,:]
train_sol_5 = readdlm("Bursty/Control_rate_Inference/Control_tau_fixed_otherrand/data/set1/120-120.csv",',')[2:end,:]

plot(0:N,vec(train_sol_1),lw=3,label="car 1")
plot!(0:N,vec(train_sol_2),lw=3,label="car 2")
plot!(0:N,vec(train_sol_3),lw=3,label="car 3")
plot!(0:N,vec(train_sol_4),lw=3,label="car 4")
plot!(0:N,vec(train_sol_5),lw=3,label="car 5")



