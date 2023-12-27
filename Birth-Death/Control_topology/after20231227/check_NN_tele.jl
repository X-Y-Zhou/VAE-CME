using Plots
using LinearAlgebra, Distributions, DifferentialEquations

P_split = [p_0;p_1]
N = 100

function f(NN)
    NN1 = NN[1:N]
    NN2 = NN[N+1:2*N]
    [-sigma_on*P_split[1] + NN1[1]*P_split[2] + sigma_off*P_split[N+1];
    [(-sigma_on-NN1[i-1])*P_split[i] + NN1[i]*P_split[i+1] + sigma_off*P_split[i+N] for i in 2:N-1];
    (-sigma_on-NN1[N-1])*P_split[N] + sigma_off*P_split[2*N];
    
    sigma_on*P_split[1] + (-sigma_off-rho_on)*P_split[N+1] + NN2[1]*P_split[N+2];
    [sigma_on*P_split[i-N] + rho_on*P_split[i-1] + (-sigma_off-rho_on-NN2[i-N-1])*P_split[i]+NN2[i-N]*P_split[i+1] for i in (N+2):(2*N-1)];
    sigma_on*P_split[N] + rho_on*P_split[2*N-1]+(-sigma_off-rho_on-NN2[N-1])*P_split[2*N]
    ]
end

u0 = zeros(2*N)
u0[N+1] = 1.
solution = nlsolve(f, u0)
NN_solution = solution.zero
NN1 = NN_solution[1:N]
NN2 = NN_solution[N+1:2*N]
plot(NN1,label="NN1")
plot(NN2,label="NN2")



