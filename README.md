# VAE-CME
# Delay Direct Method Algorithm
Consider that $N_d$ delay reactions are ongoing at the time $t$. The delay reactions will complete at $t+T_1,\ldots,t+T_{N_d}$, where $T_1 \leq T_2,\ldots,t+T_{N_d}$. As in the derivation of Gillespie’s exact Stochastic Simulation Algorithm (SSA), $p(\tau, \nu) d\tau$ can be found from the fundamental assumption as $p(\tau, \nu) d\tau = p_0(\tau) f_\mu(t + \tau) d\tau$, where $p_0(\tau)$ is the probability that no reaction will happen in the time interval $[t,t+\tau)$. The delay effects the propensity function. So $p_0(\tau)$ comes to $\exp(-\Sigma_{j=0}^{i-1}\lambda(t+T_j)(T_{j+1}-T_j)-\lambda(t+T_i)(\tau-T_{i})),~~\tau \in [T_i, T_{i+1}), ~~i=0,\ldots,N_d$, where the exponent assume equal to zero when $i=0$.
**The Cauchy-Schwarz Inequality**
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$
$$p(\mu|\textbf{n},t)=f_r(t + T_i)/\lambda(t + T_i),~~\mu= 1, \ldots,R, ~~\tau \in [T_i, T_{i+1}), ~~i=0,\ldots,N_d.$$

$$p(\tau|\textbf{n},t)=\lambda(t + T_i) \exp ( - \sum_{j=0}^{i-1} \lambda(t + T_j)(T_{j+1} - T_j) - \lambda(t + T_i)(\tau - T_i) ),~~\lambda(t + T_i)=\sum_{r=1}^{R} f_r(t + T_i),$$