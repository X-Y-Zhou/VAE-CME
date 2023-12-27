using Plots
using LinearAlgebra, Distributions, DifferentialEquations

sigma_on = 0.003
sigma_off = 0.004
rho_on = 0.3
τ = 120
N = 100

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
plot(p_0)
plot(p_1)

p0p1 = p_0+p_1
plot(p0p1,lw=3,line=:dash)


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

