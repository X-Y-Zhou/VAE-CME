using DelimitedFiles,Flux,Plots,DifferentialEquations,TaylorSeries
using DiffEqSensitivity

function bursty(N,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

# truncation
N = 64

a = 0.0282
b = 3.46
τ = 120
P_exat = bursty(N+1,τ)

# calculate the probabilities at 0~N
end_time = 800
exact_sol = zeros(N+1,end_time+1)
for i = 0:end_time
    if i < 120
        exact_sol[1:N+1,i+1] = bursty(N+1,i)
    else
        exact_sol[1:N+1,i+1] = bursty(N+1,120)
    end
end

traj = 100
train_sol = readdlm("machine-learning/ode/bursty/Reduce_sample_size/$traj.csv",',')[2:end,:]


# model initialization
model = Chain(Dense(N+1, 25, tanh), Dense(25, N),x -> 0.03.*x .+ [i/120 for i in 1:N], x->relu.(x))
params, re = Flux.destructure(model)
ps = Flux.params(params);
params

# define the CME
function CME(du, u, p, t)
    NN = re(p)(u);
    du[1] = (-a*b/(1+b))*u[1] + NN[1]*u[2];
    for i in 2:N
            du[i] =  (-a*b/(1+b) - NN[i-1])*u[i] + NN[i]*u[i+1];
            for j in 1:i-1
                    du[i] += (a*(b/(1+b))^j/(1+b)) * u[i-j]
            end
    end
    du[N+1] = (-a*b/(1+b) - NN[N])*u[N+1];
    for j in 1:N
            du[N+1] += (a*(b/(1+b))^j/(1+b)) * u[N+1-j]
    end
end

# initialize the ODE solver
u0 = [1.; zeros(N)]
tf = 800; #end time
tspan = (0, tf);
saveat = [1:5:120;140:20:800]
problem = ODEProblem(CME, u0, tspan, params);
solution = solve(problem,Tsit5(),u0=u0,p=params,saveat=saveat)

# define the objective function
function loss_adjoint(p)
    sol = Array(solve(ODEProblem(CME, u0, tspan, p), Tsit5(), u0=u0, p=p, saveat=saveat));
    # calculate MSE at the selected time between train data and fitted data
    Flux.mse(sol, train_sol[:,saveat.+1])
end
loss_adjoint(params)
grads = gradient(()->loss_adjoint(params), ps)

epochs_all = 0
#0.05*50,0.025*70
lr = 0.08
opt= ADAM(lr);
epochs = 2
epochs_all = epochs_all + epochs
print("learning rate = ",lr)
@time for epoch in 1:epochs
    grads = gradient(()->loss_adjoint(params), ps)
    Flux.update!(opt,ps,grads)
    u0 = [1.; zeros(N)];
    use_time=800;
    time_step = 1.0; 
    tspan = (0.0, use_time);
    problem = ODEProblem(CME, u0, tspan,params);
    solution = Array(solve(problem, Tsit5(), u0=u0, p=params,saveat=0:time_step:Int(use_time)))
    print(epoch,"\n",Flux.mse(solution,exact_sol),"\n")
end

u0 = [1.; zeros(N)];
use_time=800;
time_step = 1.0; 
tspan = (0.0, use_time);
problem = ODEProblem(CME, u0, tspan,params);
solution = Array(solve(problem, Tsit5(), u0=u0, p=params,saveat=0:time_step:Int(use_time)))

#check mean
mean_exact = [sum([j for j=0:N].*exact_sol[:,i]) for i=1:size(exact_sol,2)]
mean_trained = [sum([j for j=0:N].*solution[:,i]) for i=1:size(solution,2)]
plot(mean_trained,linewidth = 3,label="NN-CME",xlabel = "# t", ylabel = "mean value")
plot!(mean_exact,label="exact",linewidth = 3,line=:dash,legend=:bottomright)

#check probabilities
function plot_distribution(time_choose)
    p=plot(0:N,solution[:,time_choose],linewidth = 3,label="VAE-CME",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N,exact_sol[:,time_choose+1],linewidth = 3,label="exact",title=join(["t=",time_choose]),line=:dash)
    return p
end

function plot_all()
    p1 = plot_distribution(27)
    p2 = plot_distribution(47)
    p3 = plot_distribution(67)
    p4 = plot_distribution(77)
    p5 = plot_distribution(87)
    p6 = plot_distribution(97)
    p7 = plot_distribution(107)
    p8 = plot_distribution(120)
    p9 = plot_distribution(200)
    p10 = plot_distribution(300)
    p11 = plot_distribution(500)
    p12 = plot_distribution(800)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,size=(1200,900))
end
plot_all()

Flux.mse(solution,exact_sol)

#write params
using DataFrames,CSV
params
df = DataFrame(params = params)
CSV.write("machine-learning/ode/bursty/Reduce_sample_size/params_MLP_$traj.csv",df)

#read params
using CSV,DataFrames
df = CSV.read("machine-learning/ode/bursty/Reduce_sample_size/params_MLP_$traj.csv",DataFrame)
params = df.params
ps = Flux.params(params);

tra = [100,200,300,1000,3000,1e4,1e5]
x = log10.(tra)
y = [2.70E-05,4.83E-06,2.31E-06,1.07E-06,6.55E-07,4.47E-07]
plot(x,y)
scatter!(x,y)