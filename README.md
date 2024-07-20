# VAE-CME

This repository deposits the Julia codes and data associated with the effeciency test presented in:
**An efficient approach for modeling Non-Markovian gene expression dynamics with Variational Autoencoders.**

**File descriptions:**
- "Fig2b" and "Fig3b" includes the training data, training results and parameters trained by VAE of four models, which are Birth Death model, Bursty model, Telegraph model and Oscillation model. 
- "Fig5b" describes how VAE predicts distributions for reaction network with different delay mechanism, different kinetic parameters and different topology.

**Requirements:**

- Julia >= 1.6.5
- Flux v0.12.8
- DifferentialEquations v7.0.0
- DiffEqSensitivity v6.66.0
- Zygote v0.6.33

**The method is well described in:**

* X. Zhou _et. al._ [An efficient approach for modeling Non-Markovian gene expression dynamics with Variational Autoencoders]().
