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

# Solve CME in the steady state
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

# Exact solution of birth-death model
function birth_death_delay(N,ρ,τ)
    distribution = Poisson(ρ*τ)
    P = zeros(N)
    for i=1:N
        P[i] = pdf(distribution,i-1)
    end
    return P
end;

# Exact solution of bursty model
function bursty_delay(N,a,b,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;