using TaylorSeries,LinearMaps,StatsBase,Distributions
using LinearAlgebra,DelimitedFiles

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