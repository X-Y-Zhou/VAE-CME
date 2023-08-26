using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots,CSV,DataFrames
using BlackBoxOptim

include("../../../utils.jl")

#exact solution
function bursty(N,a,b,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

a = 0.0282;
b = 3.46;
τ = 120;
N = 64

a_list = [0.0082,0.0132,0.0182,0.0232,0.0282]
b_list = [1.46,1.96,2.46,2.96,3.46]
l_ablist = length(a_list)*length(b_list)

ab_list = [[a_list[i],b_list[j]] for i=1:length(a_list) for j=1:length(b_list)]
train_sol = [bursty(N,ab_list[i][1],ab_list[i][2],τ) for i=1:l_ablist]

# Model 
latent_size = 10;
encoder = Chain(Dense(N, 20,tanh),Dense(20, latent_size * 2));
decoder = Chain(Dense(latent_size, 20),Dense(20 , N-1),x->0.03.* x.+[i/τ  for i in 1:N-1],x ->relu.(x));

params1, re1 = Flux.destructure(encoder);
params2, re2 = Flux.destructure(decoder);
ps = Flux.params(params1,params2);

function f1!(x,p1,p2,a,b,ϵ)
    h = re1(p1)(x)
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    NN = re2(p2)(z)
    return vcat(-a*b/(1+b)*x[1]+NN[1]*x[2],[sum(a*(b/(1+b))^(i-j)/(1+b)*x[j] for j in 1:i-1) - 
            (a*b/(1+b)+NN[i-1])*x[i] + NN[i]*x[i+1] for i in 2:N-1],sum(x)-1)
end

#solve P
P_0_distribution = NegativeBinomial(a*τ, 1/(1+b));
P_0_list = [[pdf(NegativeBinomial(ab_list[i][1]*τ, 1/(1+ab_list[i][2])),j) for j=0:N-1] for i=1:l_ablist]

ϵ = zeros(latent_size)
sol(p1,p2,a,b,ϵ,P0) = nlsolve(x->f1!(x,p1,p2,a,b,ϵ),P0).zero
solution = sol(params1,params2,a,b,ϵ,P_0_list[25])

sample_size = 1e4
solution = set_one.(solution)
# log_value = [log.(solution[i]) for i=1:use_time+1]
log_value = log.(train_sol[:,i])


# SSA data
SSA_timepoints = counts(round.(Int, SSA_data[i,:]))[1:N+1]

# # exact data 
# SSA_timepoints = [round.(Int, train_sol[:,i].*sample_size) for i=1:use_time+1]

logp_x_z = sum(SSA_timepoints[i].*log_value[i])/sample_size


# a b τ
kinetic_params = [a,b,τ]

function LogLikelihood(kinetic_params)
    a = kinetic_params[1]
    b = kinetic_params[2]

    df = CSV.read("Bursty/Control_rate_Inference/control_kinetic/params_ck.csv",DataFrame)
    params1 = df.params1
    params2 = df.params2[1:length_2]

    ϵ = zeros(latent_size)
    P_0 = [pdf(NegativeBinomial(a*τ, 1/(1+b)),j) for j=0:N-1]
    solution = sol(params1,params2,a,b,ϵ,P_0)

    solution = set_one.(solution)
    log_value = log.(solution[i])
    loglikelihood_value = -sum(SSA_timepoints[i].*log_value[i])/sample_size
 
    return loglikelihood_value
end