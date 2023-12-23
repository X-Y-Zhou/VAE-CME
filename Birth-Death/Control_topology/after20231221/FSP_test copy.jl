using Plots
using LinearAlgebra, Distributions, SparseArrays, DifferentialEquations

# Define kinetic parameters
d=1
ρ=10
σon = 0.02
σoff = 0.05
τ=2
N=60

# Define transition matrix without delay effect terms
M = zeros(2*N^2,2*N^2) 
for i = 1 : N
    M[(i-1)*N+1:i*N, (i-1)*N+1:i*N] = -spdiagm(0 => σon*ones(N)) - spdiagm(d*(i-1)*ones(N))
    M[(i-1)*N+1:i*N, N^2+(i-1)*N+1:N^2+i*N] = spdiagm( 0 => σoff*ones(N))
    M[N^2+(i-1)*N+1:N^2+i*N, (i-1)*N+1:i*N] = spdiagm(0 => σon*ones(N))
    M[N^2+(i-1)*N+1:N^2+i*N, N^2+(i-1)*N+1:N^2+i*N] = -spdiagm(0 => σoff*ones(N)) - spdiagm(d*(i-1)*ones(N)) - spdiagm(0 => ρ*vcat(ones(N-1),0)) + spdiagm(-1 => ρ*ones(N-1))
end

for i = 1 : N-1
    M[(i-1)*N+1:i*N, (i-1)*N+1+N:i*N+N] = spdiagm(d*i*ones(N))
    M[N^2+(i-1)*N+1:N^2+i*N, N^2+(i-1)*N+1+N:i*N+N^2+N] = spdiagm(d*i*ones(N))
end

# Solve Delay CME in time [0,τ]
function CME_tau!(du,u,p,t)
    ρ,τ,N=p
    du[1:end] = M*u
end

p = (ρ,τ,N)
u0 = zeros(2*N^2)
# Initial condition -- gene state ON
u0[N^2+1] = 1
tspan = (0.0, τ)
prob1 = ODEProblem(CME_tau!, u0, tspan, p)
sol1 = solve(prob1, Tsit5(), saveat=0.02)
u1 = sol1.u[end]
#u1m = reshape(u1[1:N^2]+u1[1+N^2:end],(N,N))
#upn = sum(u1m,dims = 2)
#plot(collect(0:N-1),upn)

# Define initial conditions for CME in time [τ,T]
u_temp = zeros(2*N)
# Initial condition -- gene state ON
u_temp[N+1] = 1
u_main = vcat(u1,u_temp)

# Define transition matrix for Q's CME
A1 = - spdiagm(0 => σon*ones(N)) - spdiagm(0 => d*collect(0:N-1)) + spdiagm(1 => d*collect(1:N-1))
A2 =  spdiagm(0 => σoff*ones(N)) 
A3 = spdiagm(0 => σon*ones(N)) 
A4 = - spdiagm(0 => σoff*ones(N)) - spdiagm(0 => d*collect(0:N-1)) + diagm(1 => d*collect(1:N-1)) - spdiagm(0 => ρ*vcat(ones(N-1),0)) + spdiagm(-1 => ρ*ones(N-1))
A = zeros(2*N,2*N)
A[1:N,1:N] = A1
A[1:N,N+1:2*N] = A2
A[N+1:2*N,1:N] = A3
A[N+1:2*N,N+1:2*N] = A4

# Define transition matrix for \bar{P}'s CME
B1 = - spdiagm(0 => σon*ones(N)) 
B2 =  spdiagm(0 => σoff*ones(N)) 
B3 = spdiagm(0 => σon*ones(N)) 
B4 = - spdiagm(0 => σoff*ones(N))  - spdiagm(0 => ρ*vcat(ones(N-1),0)) + spdiagm(-1 => ρ*ones(N-1))
B = zeros(2*N,2*N)
B[1:N,1:N] = B1
B[1:N,N+1:2*N] = B2
B[N+1:2*N,1:N] = B3
B[N+1:2*N,N+1:2*N] = B4

# Solve \bar{P}_0(n) and \bar{P}_1(n)
function CME_aux!(du,u,p,t)
    ρ,τ,N=p
    du[1:end] = B*u
end
u_aux = zeros(2*N)
# Initial condition -- gene state ON (always true)
u_aux[N+1] = 1.
tspan = (0.0, τ)
prob2 = ODEProblem(CME_aux!, u_aux, tspan, p)
sol2 = solve(prob2, Tsit5(), saveat=0.02)
pn0 = sol2.u[end][1:N]
pn1 = sol2.u[end][N+1:end]
#plot!(pn0+pn1)
pn01 = vcat(0,pn0[1:end-1])
pn11 = vcat(0,pn1[1:end-1])

# Solve for joint distribution in time [τ,T]
function CME_main!(du,u,p,t)
    ρ,τ,N=p
    v = u[N^2*2+1+N:end]
    v1 = vcat(0,v[1:end-1])
    d_term1 = reshape(pn0*v1'-pn01*v',(N^2,1))
    d_term2 = reshape(pn1*v1'-pn11*v',(N^2,1))
    du[1:N^2*2] = M*u[1:N^2*2] + ρ * vcat(d_term1,d_term2)
    du[N^2*2+1:end] = A*u[N^2*2+1:end]
end

tspan = (0.0, 10)
prob3 = ODEProblem(CME_main!, u_main, tspan, p)
sol3 = solve(prob3, Tsit5(), saveat=1)

T = 4 # Absolute time 6
tsp = trunc(Int,T/0.02+1)
tsp = 4
u_int = sol3.u[tsp][1:N^2*2]
u_intm = reshape(u_int[1:N^2]+u_int[N^2+1:2*N^2],(N,N))
#u1m = reshape(u1[1:N^2]+u1[1+N^2:end],(N,N))
u_intn = sum(u_intm,dims = 2)
plot(collect(0:N-1),u_intm[:,16])