using Plots
tra = [100,300,1000,3000]
x = log10.(tra)
y_MLP = [4.01e-5,4.83E-06,2.31E-06,1.07E-06]
y_VAE = [3.45E-05,4.30E-06,2.47E-06,1.41E-06]

plot(x,y_MLP,label="MLP")
scatter!(x,y_MLP,label=false,color=:blue,xlabel="log10 SSA trajectories")
plot!(x,y_VAE,label="VAE")
scatter!(x,y_VAE,label=false,color=:green,ylabel="MSE")

savefig("Bursty/Reduce_sample_size/results/compare_MLP_VAE.pdf")