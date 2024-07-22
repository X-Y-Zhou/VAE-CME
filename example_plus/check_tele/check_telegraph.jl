# Parallel
using Distributed,Pkg
addprocs(3)

# Import packages
@everywhere using Flux, DiffEqSensitivity, DifferentialEquations
@everywhere using Distributions, Distances,Random
@everywhere using DelimitedFiles, Plots
@everywhere include("../../utils.jl")

# Define time delay and truncation
@everywhere τ = 30
@everywhere N = 120

# Define kinetic parameters
@everywhere ps_matrix_tele = readdlm("example_plus/check_tele/ps_tele_check.txt")
@everywhere sigma_on_list = ps_matrix_tele[1,:]
@everywhere sigma_off_list = ps_matrix_tele[2,:]
@everywhere rho_on_list = ps_matrix_tele[3,:]
@everywhere rho_off = 0.0
@everywhere gamma= 0.0
@everywhere batchsize_tele = size(ps_matrix_tele,2)

# Load check data 
check_sol0 = readdlm("example_plus/check_tele/SSA_proba/tele_SSA_Attr=0.0.txt")   # Attrtibute = 0      τ ~ Normal(30,0)
check_sol1 = readdlm("example_plus/check_tele/SSA_proba/tele_SSA_Attr=0.25.txt")  # Attrtibute = 0.25   τ ~ Normal(30,2.5)
check_sol2 = readdlm("example_plus/check_tele/SSA_proba/tele_SSA_Attr=0.50.txt")  # Attrtibute = 0.5    τ ~ Normal(30,5)
check_sol3 = readdlm("example_plus/check_tele/SSA_proba/tele_SSA_Attr=0.75.txt")  # Attrtibute = 0.75   τ ~ Normal(30,7.5)
check_sol4 = readdlm("example_plus/check_tele/SSA_proba/tele_SSA_Attr=1.0.txt")   # Attrtibute = 1.0    τ ~ Normal(30,10)

# Model initialization
@everywhere latent_size = 2;
@everywhere encoder = Chain(Dense(N, 10,tanh),Dense(10, latent_size * 2));
@everywhere decoder_1 = Chain(Dense(latent_size+1, 10),Dense(10 , N-1),x -> x.+[i/τ  for i in 1:N-1],x ->relu.(x));
@everywhere decoder_2 = Chain(decoder_1[1],decoder_1[2],decoder_1[3],decoder_1[4]);

@everywhere params1, re1 = Flux.destructure(encoder);
@everywhere params2, re2_1 = Flux.destructure(decoder_1);
@everywhere       _, re2_2 = Flux.destructure(decoder_2);
@everywhere ps = Flux.params(params1,params2);

# Define the CME in the steady state
@everywhere function f_tele!(x,p1,p2,ϵ,sigma_on,sigma_off,rho_on,Attribute)
    h = re1(p1)(x[1:N])
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,Attribute)
    NN1 = re2_1(p2)(z)

    h = re1(p1)(x[N+1:2*N])
    μ, logσ = split_encoder_result(h, latent_size)
    z = reparameterize.(μ, logσ, ϵ)
    z = vcat(z,Attribute)
    NN2 = re2_1(p2)(z)
    
    return vcat((-sigma_on-rho_off)*x[1] + (-gamma+NN1[1])*x[2] + sigma_off*x[N+1],
                [rho_off*x[i-1] + (-sigma_on-rho_off+(i-1)*gamma-NN1[i-1])*x[i] + (-i*gamma+NN1[i])*x[i+1] + sigma_off*x[i+N] for i in 2:N-1],
                rho_off*x[N-1] + (-sigma_on-rho_off+N*gamma-NN1[N-1])*x[N] + sigma_off*x[2*N],
                
                sigma_on*x[1] + (-sigma_off-rho_on)*x[N+1] + (-gamma+NN2[1])*x[N+2],
                [sigma_on*x[i-N] + rho_on*x[i-1] + (-sigma_off-rho_on+(i-N-1)*gamma -NN2[i-N-1])*x[i] + (-(i-N)*gamma+NN2[i-N])*x[i+1] for i in (N+2):(2*N-1)],
                sum(x)-1)
end

# Define the CME solver
@everywhere sol_tele(p1,p2,ϵ,sigma_on,sigma_off,rho_on,P_0,Attrtibute) = nlsolve(x->f_tele!(x,p1,p2,ϵ,sigma_on,sigma_off,rho_on,Attrtibute),P_0).zero
@everywhere function solve_tele(sigma_on,sigma_off,rho_on,p1,p2,ϵ,Attrtibute)
    P_0_distribution = Poisson(rho_on*τ*sigma_on)
    P_0 = [pdf(P_0_distribution,j) for j=0:N-1]
    P_0_split = [P_0*sigma_on/(sigma_on+sigma_off);P_0*sigma_off/(sigma_on+sigma_off)]

    solution = sol_tele(p1,p2,ϵ,sigma_on,sigma_off,rho_on,P_0_split,Attrtibute)
    solution = solution[1:N]+solution[N+1:2*N]
    return solution
end

# Read trained VAE parameters
using CSV,DataFrames
df = CSV.read("example_plus/params_trained.csv",DataFrame)
params1 = df.params1[1:length(params1)]
params2 = df.params2[1:length(params2)]

# Solve the CME
ϵ = zeros(latent_size)
Attrtibute = 0
@time solution_tele0 = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
mse_tele0 = Flux.mse(solution_tele0,check_sol0)

Attrtibute = 0.25
@time solution_tele1 = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
mse_tele1 = Flux.mse(solution_tele1,check_sol1)

Attrtibute = 0.5
@time solution_tele2 = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
mse_tele2 = Flux.mse(solution_tele2,check_sol2)

Attrtibute = 0.75
@time solution_tele3 = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
mse_tele3 = Flux.mse(solution_tele3,check_sol3)

Attrtibute = 1.0
@time solution_tele4 = hcat(pmap(i->solve_tele(sigma_on_list[i],sigma_off_list[i],rho_on_list[i],params1,params2,ϵ,Attrtibute),1:batchsize_tele)...);
mse_tele4 = Flux.mse(solution_tele4,check_sol4)

# Plot probability distribution
function plot_distribution(set)
    plot(0:N-1,solution_tele4[:,set],linewidth = 3,label="VAE-CME",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(0:N-1,check_sol4[:,set],linewidth = 3,label="exact",title=join([round.(ps_matrix_tele[:,set],digits=4)]),line=:dash)
end

function plot_channel()
    p1 = plot_distribution(1)
    p2 = plot_distribution(2)
    p3 = plot_distribution(3)
    p4 = plot_distribution(4)
    p5 = plot_distribution(5)
    p6 = plot_distribution(6)
    plot(p1,p2,p3,p4,p5,p6,layouts=(3,2),size=(600,900))
end
plot_channel()
savefig("example_plus/check_tele/results/Attr=1.0.svg")

writedlm("example_plus/check_tele/pre_proba/tele_pre_Attr=0.0.txt",solution_tele0)
# writedlm("example_plus/check_tele/pre_proba/tele_pre_Attr=0.25.txt",solution_tele1)
# writedlm("example_plus/check_tele/pre_proba/tele_pre_Attr=0.5.txt",solution_tele2)
# writedlm("example_plus/check_tele/pre_proba/tele_pre_Attr=0.75.txt",solution_tele3)
writedlm("example_plus/check_tele/pre_proba/tele_pre_Attr=1.0.txt",solution_tele4)







