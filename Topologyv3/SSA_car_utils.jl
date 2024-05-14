using Distributions,Plots

# The input dist is the τ of the system, which can be either a probability distribution or a fixed value; 
# if a fixed value is input, it degenerates into the case where τ remains unchanged
function generate_velo(dist,L)
    if isa(dist, Distribution)
        τ = rand(dist)
        velo = L/τ
    else
        τ = dist
        velo = L/τ
    end
    return velo
end

# SSA bd
# ρ 0 -> N will trigger N => 0 after time τ 
function car_event_bd(tmax,saveat,ρ,dist,L)
    velo_list = []
    location_list = []
    t = 0
    t_list = [0.]
    n_list = [0]
    while t<tmax 
        min_t = 0
        Δ = rand(Exponential(1/ρ))

        channel = [Δ,Inf,Inf] # 1.car enter 2.car leave 3.car meet

        # generate leave chanel 
        if length(location_list) != 0
            channel[end-1] = (L-location_list[1])/velo_list[1]
        end

        # generate meet chanel
        if length(location_list)>1
            meet_channel = [(location_list[i]-location_list[i+1])/(velo_list[i+1]-velo_list[i]) for i=1:length(location_list)-1]
            if length(filter(x->x>0,meet_channel))>0
                channel[end] = minimum(filter(x->x>0,meet_channel))
            end
        end

        min_t,event = findmin(channel)

        if length(t_list) != 1
            location_list = location_list + velo_list.*min_t
        end
        
        if event == 1 # a car enter
            vv = generate_velo(dist,L)
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

    n_saveat_list = n_list[searchsortedlast.(Ref(t_list), saveat)]
    return n_saveat_list
end

# SSA bursty
# ρ 0 -> iN will trigger N => 0 after time τ

# function process_element(j)
#     if j == 0
#         0
#     else
#         sum(rand(Geometric(1 / (1 + β))) for _ = 1:j)
#     end
# end

function car_event_bursty(tmax,saveat,α,β,dist,L)
    velo_list = []
    location_list = []
    t = 0
    t_list = [0.]
    n_list = [0]
    while t<tmax 
        min_t = 0
        Δ = rand(Exponential(1/α))

        channel = [Δ,Inf,Inf] #1.car enter 2.car leave 3.car meet

        if length(location_list) != 0
            channel[end-1] = (L-location_list[1])/velo_list[1]
        end

        if length(location_list)>1
            meet_channel = [(location_list[i]-location_list[i+1])/(velo_list[i+1]-velo_list[i]) for i=1:length(location_list)-1]
            if length(filter(x->x>0,meet_channel))>0
                channel[end] = minimum(filter(x->x>0,meet_channel))
            end
        end

        min_t,event = findmin(channel)

        if length(t_list) != 1
            location_list = location_list + velo_list.*min_t
        end
        
        if event == 1 #a car enter
            vv = generate_velo(dist,L)
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

    n_saveat_list = n_list[searchsortedlast.(Ref(t_list), saveat)]

    n_people_list=[]
    for j in n_saveat_list
        if j==0
            push!(n_people_list,0)
        else
            push!(n_people_list,sum([rand(Geometric(1/(1+β))) for i=1:j]))
        end
    end
    return n_people_list

    # n_people_list = process_element.(n_saveat_list);
    # return n_people_list

    return n_saveat_list
end

# SSA tele 
# sigma_on  G* --> G
# sigma_off G --> G*
# ρ         G --> G+N will trigger N => 0 after time τ 
function car_event_tele(tmax,saveat,sigma_on,sigma_off,ρ,dist,L)
    S = [-1  1 0; # G*
          1 -1 0; # G
          0  0 1] # N

    velo_list = []
    location_list = []

    n = [1,0,0]
    t = 0
    t_list = [0.]
    n_list = [n,]
    while t<tmax 
        min_t = 0

        u_list = rand(Uniform(0,1),3)
        fr = [n[1]*sigma_on,n[2]*sigma_off,n[2]*ρ]

        τ_list = -log.(u_list)./fr

        channel = [τ_list;Inf;Inf] 
        # 1.G* --> G 
        # 2.G --> G* 
        # 3.car enter 
        # 4.car leave 
        # 5.car meet

        if length(location_list) != 0
            channel[end-1] = (L-location_list[1])/velo_list[1]
        end

        if length(location_list)>1
            meet_channel = [(location_list[i]-location_list[i+1])/(velo_list[i+1]-velo_list[i]) for i=1:length(location_list)-1]
            if length(filter(x->x>0,meet_channel))>0
                channel[end] = minimum(filter(x->x>0,meet_channel))
            end
        end

        min_t,event = findmin(channel)

        if length(t_list) != 1
            location_list = location_list + velo_list.*min_t
        end
        
        if event < 3
            n = n .+ S[:,event]
        end

        if event == 3 # a car enter
            vv = generate_velo(dist,L)
            push!(velo_list,vv)
            push!(location_list,0.)
        end

        if event == 4 # a car leave
            filter!(x->x<L,location_list)
            velo_list = velo_list[end-length(location_list)+1:end]
        end

        if event == 5 # car meet
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

        n = [n[1],n[2],length(location_list)]
        t=t+min_t

        push!(t_list,t)
        push!(n_list,n)
    end

    n_saveat_list = n_list[searchsortedlast.(Ref(t_list), saveat)]
    n_saveat_list = map(sublist -> sublist[3], n_saveat_list)
    return n_saveat_list
end

# # exact solution of car event with Uniform distribution
# using SymPy,QuadGK,Symbolics
# function car_exact_bd(T1,T2,ρ,t,n_cars_max)
#     @variables x s
#     f(x) = x<T1 ? 0 : T1<x<T2 ? (x-T1)/(T2-T1) : 1
#     f_(x) = 1-f(x)
#     g0(x) = f_(t-x)
#     car_distribution_theory = zeros(n_cars_max)
#     car_distribution_theory[1] = exp(-ρ*quadgk(g0, 0, t, rtol=1e-3)[1])
#     for i = 2:n_cars_max
#         n = i-1
#         g(x) = exp(-ρ*(t-x))*(ρ*(t-x))^(n-1)*ρ*(1-f(t-x))*exp(-ρ*quadgk(g0, 0, x, rtol=1e-3)[1])/factorial(big(n-1))
#         car_distribution_theory[i] = quadgk(g, 0, t, rtol=1e-3)[1]
#     end
#     return car_distribution_theory
# end

# function car_exact_bursty(T1,T2,α,β,t,n_cars_max,n_people_max)
#     @variables x s
#     f(x) = x<T1 ? 0 : T1<x<T2 ? (x-T1)/(T2-T1) : 1
#     f_(x) = 1-f(x)
#     g0(x) = f_(t-x)
#     car_distribution_theory = zeros(n_cars_max)
#     car_distribution_theory[1] = exp(-α*quadgk(g0, 0, t, rtol=1e-3)[1])
#     for i = 2:n_cars_max
#         n = i-1
#         g(x) = exp(-α*(t-x))*(α*(t-x))^(n-1)*α*(1-f(t-x))*exp(-α*quadgk(g0, 0, x, rtol=1e-3)[1])/factorial(big(n-1))
#         car_distribution_theory[i] = quadgk(g, 0, t, rtol=1e-3)[1]
#     end

#     people_distribution_theory = [sum([pdf(NegativeBinomial(k, 1/(1+β)),n)*car_distribution_theory[k+1] for k=1:n_cars_max-1]) for n=0:n_people_max-1]
#     people_distribution_theory[1]=sum([pdf(NegativeBinomial(k, 1/(1+β)),0)*car_distribution_theory[k+1] for k=1:n_cars_max-1])+car_distribution_theory[1]
#     return people_distribution_theory
# end

# T1 = 20
# T2 = 120
# dist_temp = Uniform(T1,T2)
# f(x) = x<T1 ? 0 : T1<=x<=T2 ? (x-T1)/(T2-T1) : 1
# g(x) = cdf(dist_temp,x)


# x0 = 0:200
# fx = f.(x0)
# gx = g.(x0)

# Flux.mse(fx,gx)

# exact solution of car event with any distribution
using QuadGK,Symbolics
function car_exact_bd(dist,ρ,t,n_cars_max)
    @variables x s
    f(x) = cdf(dist,x)
    f_(x) = 1-f(x)
    g0(x) = f_(t-x)
    car_distribution_theory = zeros(n_cars_max)
    car_distribution_theory[1] = exp(-ρ*quadgk(g0, 0, t, rtol=1e-3)[1])
    for i = 2:n_cars_max
        n = i-1
        g(x) = exp(-ρ*(t-x))*(ρ*(t-x))^(n-1)*ρ*(1-f(t-x))*exp(-ρ*quadgk(g0, 0, x, rtol=1e-3)[1])/factorial(big(n-1))
        car_distribution_theory[i] = quadgk(g, 0, t, rtol=1e-3)[1]
    end
    return car_distribution_theory
end

function car_exact_bursty(dist,α,β,t,n_cars_max,n_people_max)
    @variables x s
    f(x) = cdf(dist,x)
    f_(x) = 1-f(x)
    g0(x) = f_(t-x)
    car_distribution_theory = zeros(n_cars_max)
    car_distribution_theory[1] = exp(-α*quadgk(g0, 0, t, rtol=1e-3)[1])
    for i = 2:n_cars_max
        n = i-1
        g(x) = exp(-α*(t-x))*(α*(t-x))^(n-1)*α*(1-f(t-x))*exp(-α*quadgk(g0, 0, x, rtol=1e-3)[1])/factorial(big(n-1))
        car_distribution_theory[i] = quadgk(g, 0, t, rtol=1e-3)[1]
    end

    people_distribution_theory = [sum([pdf(NegativeBinomial(k, 1/(1+β)),n)*car_distribution_theory[k+1] for k=1:n_cars_max-1]) for n=0:n_people_max-1]
    people_distribution_theory[1]=sum([pdf(NegativeBinomial(k, 1/(1+β)),0)*car_distribution_theory[k+1] for k=1:n_cars_max-1])+car_distribution_theory[1]
    return people_distribution_theory
end
