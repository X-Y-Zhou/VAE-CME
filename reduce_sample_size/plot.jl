using Plots
tra = [100,300,1000,3000,10000]
x = log10.(tra)
y_MLP = [0.000184462,0.000182487,0.000151652,9.88355E-05,9.5466E-05]
y_VAE = [0.000123495,9.2042E-05,8.30677E-05,7.82634E-05,8.19834E-05]

plot(x,y_MLP,label="MLP",ylabel="MSE_X",xlabel="log10 SSA")
scatter!(x,y_MLP,label=false,color="blue")
plot!(x,y_VAE,label="VAE")
scatter!(x,y_VAE,label=false,color="green")
savefig("reduce_sample_size/compare_mse_X.svg")