using Plots
tra = [100,300,1000,3000,10000]
x = log10.(tra)
y_MLP = [0.000177596,0.000123008,0.00010839,9.62599E-05,8.90072E-05]
y_VAE = [9.4924E-05,8.59674E-05,8.03695E-05,7.60701E-05,7.81849E-05]

plot(x,y_MLP,label="MLP",ylabel="MSE_X",xlabel="log10 SSA")
scatter!(x,y_MLP,label=false,color="blue")
plot!(x,y_VAE,label="VAE")
scatter!(x,y_VAE,label=false,color="green")
savefig("reduce_sample_size/compare_mse_X.svg")