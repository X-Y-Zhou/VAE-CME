using TaylorSeries,LinearMaps,StatsBase,Distributions
using LinearAlgebra,DelimitedFiles
using NLsolve,Zygote,IterativeSolvers
using Zygote:@adjoint
using CSV,DataFrames

# Convert a vector to probability distributions
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

@adjoint nlsolve(f, x0;kwargs...) =
    let result = nlsolve(f,x0;kwargs...)
        result, function(vresult)
            dx = vresult[].zero
            x = result.zero
            _,back_x = Zygote.pullback(f,x)

            JT(df) =back_x(df)[1]
            # solve JT*df =-dx
            L = LinearMap(JT, length(x0))
            df = gmres(L,-dx)

            _,back_f = Zygote.pullback(f -> f(x),f)
            return (back_f(df)[1],nothing,nothing)
        end
    end

# Get μ and logσ
function split_encoder_result(X, n_latent::Int64)
    μ = X[1:n_latent, :]
    logσ = X[(n_latent + 1):(n_latent * 2), :]
    return μ, logσ
end

# Reparameterize
function reparameterize(μ :: T, logσ :: T, ϵ) where {T}
    return  ϵ * exp(logσ * 0.5f0) + μ
end

# Get first derivative
function Derivative_approxi(vec::Vector)
    return [abs(vec[i+1]-vec[i]) for i=1:length(vec)-1]
end

# Calculate mean value according to the distribution P
P2mean(P) = [P[i] * (i-1) for i in 1:length(P)] |> sum

# Calculate variance var
P2var(P) = ([P[i] * (i-1)^2 for i in 1:length(P)] |> sum) - P2mean(P)^2

# Calculate second moment sm
P2sm(P) = [P[i] * (i-1)^2 for i in 1:length(P)] |> sum

# NN function
function f_NN(x,l,m,n,o)
    return l*x^m/(n+x^o)
end;

# exact solution of birth-death model
function birth_death_delay(N,ρ,τ)
    distribution = Poisson(ρ*τ)
    P = zeros(N)
    for i=1:N
        P[i] = pdf(distribution,i-1)
    end
    return P
end;

# exact solution of bursty model
function bursty_delay(N,a,b,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

function NN_bursty(N,a,b,τ)
    f(u) = a*b*u/(1-b*(u-1))
    g(u) = exp(a*b*τ*(u-1)/(1-b*(u-1)))
    fg(u) = f(u)*g(u)

    taylorexpand_fg = taylor_expand(x->fg(x),0,order=N)
    taylorexpand_g = taylor_expand(x->g(x),0,order=N)
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand_fg[j]/taylorexpand_g[j]
    end
    return P
end

# normalization
function set_one(vec)
    vec = abs.(vec)
    while sum(vec)>1
        vec = vec./sum(vec)
    end
    return vec
end

using DifferentialEquations
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