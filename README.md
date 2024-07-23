# VAE-CME

This repository deposits the Julia codes and data associated with the effeciency test presented in:
**Efficient solver of chemical master equation for stochastic biochemical kinetics: a variational autoencoder approach.**

**File descriptions:**
- "Fig3def" and "Fig4b" includes the training data, training results and parameters trained by VAE of four models, which are Birth Death model, Bursty model, Telegraph model and Oscillation model. 
- "Fig5bcd_Fig6b" describes how VAE predicts distributions for reaction systems with different delay mechanisms, different kinetic parameters and different topologies.
- "example_plus" describes an example that similiar to the example in "Fig5bcd_Fig6b". The only difference is that the time delay $\tau$ in "example_plus" follows the normal distribution while the time delay $\tau$ in "Fig5bcd_Fig6b" follows the uniform distribution.

**Requirements:**

- Julia >= 1.6.5
- Flux v0.12.8
- DifferentialEquations v7.0.0
- DiffEqSensitivity v6.66.0
- Zygote v0.6.33

