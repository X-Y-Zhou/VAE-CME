using Plots,NLsolve,DelimitedFiles,DifferentialEquations
using LinearAlgebra, Distributions,Random

function tele_delay(N,sigma_on,sigma_off,rho_on,τ)
    function CME_ba(du,u,p,t)
        for i=1:N
            du[i] = -sigma_on*u[i] + sigma_off*u[i+N]
        end

        du[N+1] = sigma_on*u[1] + (-sigma_off-rho_on)*u[N+1];
        for i=N+2:2*N-1
        du[i] = sigma_on*u[i-N] + rho_on*u[i-1] + (-sigma_off-rho_on)*u[i]
        end
        du[2*N] = sigma_on*u[N]+ rho_on*u[2*N-1] + (-sigma_off-rho_on)*u[2*N]
    end

    u0 = zeros(2*N)
    u0[N+1] = 1.
    tspan = (0.0, τ)
    prob1 = ODEProblem(CME_ba, u0, tspan)
    sol1 = solve(prob1, Tsit5(), saveat=1)

    P0ba = sol1.u[end][1:N]
    P1ba = sol1.u[end][N+1:2*N]
    P0ba = vcat(0,P0ba)
    P1ba = vcat(0,P1ba)

    γ = sigma_on/(sigma_on+sigma_off)
    function f(P_split)
        [[-sigma_on*P_split[i] + sigma_off*P_split[i+N]+ rho_on*γ*(P0ba[i+1]-P0ba[i]) for i=1:N]
        
        sigma_on*P_split[1] + (-sigma_off-rho_on)*P_split[N+1]+rho_on*γ*(P1ba[2]-P1ba[1]);
        [sigma_on*P_split[i-N] + rho_on*P_split[i-1] + (-sigma_off-rho_on)*P_split[i]+rho_on*γ*(P1ba[i+1-N]-P1ba[i-N]) for i=N+2:2*N-1];
        sigma_on*P_split[N]+ rho_on*P_split[2*N-1] + (-sigma_off-rho_on)*P_split[2*N] + rho_on*γ*(P1ba[N+1]-P1ba[N])
        ]
    end

    solution = nlsolve(f, [1.;zeros(2*N-1)]).zero
    p_0 = solution[1:N]
    p_1 = solution[N+1:2*N]
    p0p1 = p_0+p_1
    return p_0,p_1,p0p1
end

sigma_on = 0.8
sigma_off = 8
rho_on = 30
τ = 10
N = 120

# set = 43
# sigma_on,sigma_off,rho_on = ps_list[set]
p0p1 = tele_delay(N,sigma_on,sigma_off,rho_on,τ)[3]
plot(0:N-1,p0p1,lw=3,title=join([round.([sigma_on,sigma_off,rho_on],digits=4)]))
# plot(p0p1,lw=3,title=join([round.(ps_list[set],digits=4)]))
# plot(0:N-1,birth_death(N, rho_on, τ))

mean = P2mean(p0p1)
var = P2var(p0p1)
p_value = 1-mean/var
r_value = mean*(1-p_value)/p_value

distribution_NB = NegativeBinomial(8,1-p_value)
P_NB = [pdf(distribution_NB,i) for i=0:N-1]
plot(0:N-1,P_NB,lw=3)

p_value*r_value/(1-p_value)^2


# multi params
using Plots,NLsolve
using LinearAlgebra, Distributions, DifferentialEquations

τ = 10
N = 120

ps_list = readdlm("Topologyv4/tele/data/ps_telev2.txt")
batchsize = size(ps_list,2)
matrix_tele = zeros(N,batchsize)
matrix_tele_p0 = zeros(N,batchsize)
matrix_tele_p1 = zeros(N,batchsize)

@time for i = 1:batchsize
    print(i,"\n")
    sigma_on = ps_list[:,i][1]
    sigma_off = ps_list[:,i][2]
    rho_on = ps_list[:,i][3]
    p_0,p_1,p0p1 = tele_delay(N,sigma_on,sigma_off,rho_on,τ)

    matrix_tele_p0[:,i] = p_0
    matrix_tele_p1[:,i] = p_1
    matrix_tele[:,i] = p0p1
end

# writedlm("Topology/tele/data/matrix_telep0.csv",matrix_tele_p0)
# writedlm("Topology/tele/data/matrix_telep1.csv",matrix_tele_p1)
writedlm("Topologyv4/tele/data/matrix_telev2.csv",matrix_tele)

ps_list = readdlm("Topologyv4/tele/data/ps_telev1.txt")
matrix_tele = readdlm("Topologyv4/tele/data/matrix_telev1.csv")
N = 120
ps_list[:,20]

function plot_distribution(set)
    plot(0:N-1,matrix_tele[:,set],linewidth = 3,label="tele")
    # plot!(0:N-1,matrix_degrade[:,1,1,set],linewidth = 3,label="degrade",line=:dash,title=round.(ps_list[set],digits=4))
end
plot_distribution(10)

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
plot_channel(2)

for i = 1:5
    p = plot_channel(i)
    savefig(p,"Topologyv4/tele/data/compare/fig_$i.svg")
end

plot_channel(6)

f_γ(t) = sigma_on*(1-exp(t*(-sigma_on-sigma_off)))/(sigma_on+sigma_off)
function CME(du,u,p,t)
    for i=1:N
        du[i] = -sigma_on*u[i] + sigma_off*u[i+N] + rho_on*f_γ(t-τ)*(P0ba[i+1]-P0ba[i])
    end

    du[N+1] = sigma_on*u[1] + (-sigma_off-rho_on)*u[N+1] + rho_on*f_γ(t-τ)*(P1ba[2]-P1ba[1]);
    for i=N+2:2*N-1
       du[i] = sigma_on*u[i-N] + rho_on*u[i-1] + (-sigma_off-rho_on)*u[i] + rho_on*f_γ(t-τ)*(P1ba[i+1-N]-P1ba[i-N])
    end
    du[2*N] = sigma_on*u[N]+ rho_on*u[2*N-1] + (-sigma_off-rho_on)*u[2*N] + rho_on*f_γ(t-τ)*(P1ba[N+1]-P1ba[N])
end

u0 = zeros(2*N)
u0[N+1] = 1.
tspan = (0, 2000)
prob1 = ODEProblem(CME, u0, tspan)
solution = Array(solve(prob1, Tsit5(), saveat=1))
p0 = solution[1:N, end]
p1 = solution[N+1:end, end]
p0p1 = p0+p1
plot!(p0p1,lw=3,line=:dash)


# useless
# function CME_ba(du,u,p,t)
#     for i=1:N
#         du[i] = -sigma_on*u[i] + sigma_off*u[i+N]
#     end

#     du[N+1] = sigma_on*u[1] + (-sigma_off-rho_on)*u[N+1];
#     for i=N+2:2*N-1
#        du[i] = sigma_on*u[i-N] + rho_on*u[i-1] + (-sigma_off-rho_on)*u[i]
#     end
#     du[2*N] = sigma_on*u[N]+ rho_on*u[2*N-1] + (-sigma_off-rho_on)*u[2*N]
# end

# u0 = zeros(2*N)
# u0[N+1] = 1.
# tspan = (0.0, 200)
# prob1 = ODEProblem(CME_ba, u0, tspan)
# sol1 = solve(prob1, Tsit5(), saveat=1)

# P0ba = sol1.u[τ+1][1:N]
# P1ba = sol1.u[τ+1][N+1:2*N]

# # plot(P0ba)
# # plot(P1ba)

# P0ba = vcat(0,P0ba)
# P1ba = vcat(0,P1ba)

# γ = sigma_on/(sigma_on+sigma_off)

# function f(P_split)
#     [[-sigma_on*P_split[i] + sigma_off*P_split[i+N]+ rho_on*γ*(P0ba[i+1]-P0ba[i]) for i=1:N]
    
#     sigma_on*P_split[1] + (-sigma_off-rho_on)*P_split[N+1]+rho_on*γ*(P1ba[2]-P1ba[1]);
#     [sigma_on*P_split[i-N] + rho_on*P_split[i-1] + (-sigma_off-rho_on)*P_split[i]+rho_on*γ*(P1ba[i+1-N]-P1ba[i-N]) for i=N+2:2*N-1];
#     sigma_on*P_split[N]+ rho_on*P_split[2*N-1] + (-sigma_off-rho_on)*P_split[2*N] + rho_on*γ*(P1ba[N+1]-P1ba[N])
#     ]
# end

# solution = nlsolve(f, [1.;zeros(2*N-1)]).zero
# p_0 = solution[1:N]
# p_1 = solution[N+1:2*N]
# # plot(p_0)
# # plot(p_1)

# p0p1 = p_0+p_1
# plot(p0p1,lw=3,title=join([sigma_on," ",sigma_off," ",rho_on]))



