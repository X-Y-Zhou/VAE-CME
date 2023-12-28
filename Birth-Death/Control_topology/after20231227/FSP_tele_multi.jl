using Plots,NLsolve
using LinearAlgebra, Distributions, DifferentialEquations

sigma_on = 0.003
sigma_off = 0.004
rho_on = 0.3
τ = 120
N = 100

p_list = [[0.003,0.004,0.3],[0.003,0.008,0.3],[0.003,0.015,0.3],
          [0.0045,0.006,0.3],[0.0045,0.008,0.3],[0.0045,0.01,0.3],
          [0.006,0.0075,0.3],[0.006,0.01,0.3],[0.006,0.012,0.3],
          [0.007,0.008,0.3],[0.007,0.01,0.3],[0.006,0.015,0.3],
          [0.008,0.009,0.3],[0.008,0.015,0.3],[0.008,0.02,0.3],
          ]

p0p1_list = []
p0_list = []
p1_list = []
for p in p_list
# p = p_list[1]
print(p,"\n")
sigma_on = p[1]
sigma_off = p[2]
rho_on = p[3]
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
tspan = (0.0, 200)
prob1 = ODEProblem(CME_ba, u0, tspan)
sol1 = solve(prob1, Tsit5(), saveat=1)

P0ba = sol1.u[τ+1][1:N]
P1ba = sol1.u[τ+1][N+1:2*N]

# plot(P0ba)
# plot(P1ba)

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
push!(p0_list,p_0)
push!(p1_list,p_1)

p0p1 = p_0+p_1
push!(p0p1_list,p0p1)
end

p0p1_list
p0_list
p1_list

plot(p_0)
plot(p_1)

p0p1 = p_0+p_1
plot(p0p1,lw=3)


# f_γ(t) = sigma_on*(1-exp(t*(-sigma_on-sigma_off)))/(sigma_on+sigma_off)
# function CME(du,u,p,t)
#     for i=1:N
#         du[i] = -sigma_on*u[i] + sigma_off*u[i+N] + rho_on*f_γ(t-τ)*(P0ba[i+1]-P0ba[i])
#     end

#     du[N+1] = sigma_on*u[1] + (-sigma_off-rho_on)*u[N+1] + rho_on*f_γ(t-τ)*(P1ba[2]-P1ba[1]);
#     for i=N+2:2*N-1
#        du[i] = sigma_on*u[i-N] + rho_on*u[i-1] + (-sigma_off-rho_on)*u[i] + rho_on*f_γ(t-τ)*(P1ba[i+1-N]-P1ba[i-N])
#     end
#     du[2*N] = sigma_on*u[N]+ rho_on*u[2*N-1] + (-sigma_off-rho_on)*u[2*N] + rho_on*f_γ(t-τ)*(P1ba[N+1]-P1ba[N])
# end

# u0 = zeros(2*N)
# u0[N+1] = 1.
# tspan = (0, 200)
# prob1 = ODEProblem(CME, u0, tspan)
# sol1 = solve(prob1, Tsit5(), saveat=1)
# sol1.t
# plot(sol1.u[201])


function  plot_distribution(set)
    p=plot(0:N-1,p0p1_list[set],linewidth = 3,label="FSP",xlabel = "# of products", ylabel = "\n Probability")
    plot!(0:N-1,train_sol_end_list[set],linewidth = 3,label="SSA",line=:dash,title=join(["+-ρ=",p_list[set]]))
end
plot_distribution(1)

function plot_all()
    p1 = plot_distribution(1)
    p2 = plot_distribution(2)
    p3 = plot_distribution(3)
    p4 = plot_distribution(4)
    p5 = plot_distribution(5)
    p6 = plot_distribution(6)
    p7 = plot_distribution(7)
    p8 = plot_distribution(8)
    p9 = plot_distribution(9)
    p10 = plot_distribution(10)
    p11 = plot_distribution(11)
    p12 = plot_distribution(12)
    p13 = plot_distribution(13)
    p14 = plot_distribution(14)
    p15 = plot_distribution(15)
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
         size=(1200,1200),layout=(4,4))
    # plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,size=(1500,600),layout=(2,5))
    # plot(p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,size=(1500,900),layout=(3,5))
end
plot_all()
