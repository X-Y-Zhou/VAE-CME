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

# normalization
function set_one(vec)
    vec = abs.(vec)
    vec = vec./sum(vec)
    return vec
end