Suppose that cars enter this road according to a Possion process with rate $\lambda$.

The distance of the road is $L$.

Let $V_i$ denote the velocity value of the $i$th car enter the road

Let $T_i=L/V_i$ denote the time it would take car i to travel the road if it were empty when car $i$ arrived.

We denote

$$
G(x)=P(T_i\leq x)=P(V_i\geq L/x)
$$

So we can know the distribution of the car number $X$ on the road

$$
P(X=0)=e^{-\lambda\int^t_0\bar{G}(t-s)ds}=e^{-\lambda\int^t_0\bar{G}(u)ds} \\


P(X=n)=\int_0^t e^{-\lambda(t-y)} \frac{(\lambda(t-y))^{n-1}}{(n-1) !} \lambda\bar{G}(t-y) e^{-\lambda \int_0^y \bar{G}(t-s) d s} dy \tag{1}
$$

Note that $\bar{G}(t-s)=1-G(t-s)$

Now consider the number of people on the road when there are $Y$ people on each car.

Now we extend this theory to non Markovian models of gene expression.

According to [1], we've studied two models, which are Birth-death model and Bursty model

As for Birth-death model, it can be modeled by the following formula

$$
\emptyset\stackrel{\rho}\rightarrow N, N\stackrel{\tau}\Rightarrow\emptyset\tag{2}
$$

The producing rate $\rho$ denotes the rate $\lambda$ of Possion process. And time delay $\tau$ denotes $T_i$, the free travel time of car $i$

As for Bursty model, it can be modeled by the following formula

$$
\emptyset\stackrel{\frac{\alpha\beta^i}{(1+\beta)^{i+1}}}\longrightarrow iN,i=1,2,3,...\\ N\stackrel{\tau}\Rightarrow\emptyset\tag{3}
$$

The burtst frequency $\alpha$ denotes the rate $\lambda$ of Possion process.

$X$: number of cars, we already know distribution of $X$

$Y$: people in the car, $Y\sim Geo(\frac{1}{1+\beta})$

$Z$: people on the road

With $X$ and $Y$, we can know $Z$, consider $Z=n>0$ first,

$$
Z=\Sigma_{i=1}^X Y_i
$$

$$
P(Z=n)=\Sigma_{m=1}^\infty P(\Sigma_{i=1}^XY_i=n|X=m)P(X=m)
$$

So

$$
P(Z=n)=\Sigma_{m=1}^\infty P(\Sigma_{i=1}^m Y_i=n)P(X=m) \tag{1}
$$

Since $Y\sim Geo(\frac{1}{1+\beta})$, we can easily know that $\Sigma_{i=1}^m Y_i\sim NegativeBinomial(m,1/(1+\beta))$. Here we note that $T\sim NegativeBinomial(m,1/(1+\beta))$, and then we can simplify Eq.(1)

$$
P(Z=n)=\Sigma_{m=1}^\infty P(T=n)P(X=m),n>0,\tag{2}
$$

Now let's consider $n=0$,

$$
P(Z=0)=P(X=0)+\Sigma_{m=1}^\infty P(T=0)P(X=m),\tag{3}
$$

Eq.(2)(3) are equal to

$$
P(Z=0)=P(X=0)+\Sigma_{m=1}^\infty\theta^m P(X=m)=\Sigma_{m=0}^\infty\theta^m P(X=m)
$$

$$
P(Z=n)=\Sigma_{m=1}^\infty C^n_{n+m-1} \theta^m(1-\theta)^n P(X=m),n>0\tag{4}
$$

In Eq.(4), $\theta=\frac{1}{1+\beta}$

用VAE对同均值不同方差的 $\tau$ 进行调控，应用自由时间 $T_i=L/v$ 代替时滞 $\tau$，选取两点分布,路长度为 $L$

|      |        |        |
| :---: | :-----: | :-----: |
| $T$ | $T_1$ | $T_2$ |
| $P$ |  $p$  | $1-p$ |

$$
E=T_1p+T_2(1-p) \\ D=T_1^2p+T_2^2(1-p)-E^2
$$

$$
G(x)=

\begin{cases}

0,  0<x<T_1 \\

p,  T_1\leq x<T_2\\

1, x\geq T_2

\end{cases}
$$

此时可以知道 $V$ 的分布

|      |                  |                  |
| :---: | :---------------: | :---------------: |
| $V$ | $\frac{L}{T_2}$ | $\frac{L}{T_1}$ |
| $P$ |      $1-p$      |       $p$       |

用VAE对同均值不同方差的 $\tau$ 进行调控，用车速 自由时间 $T_i=L/v$ 代替时滞 $\tau$，选取两点分布,路长度为 $L$，

数据选取：$p=0.6$, $\alpha=\lambda=0.0282$, $\beta = 3.46$，$L=200$, 控制 $E=120$ 对以下五组数据进行测试

$T_1=\frac{L}{3},T_2=L$, $var=4266$

$T_1=\frac{3L}{10},T_2=1.05L$, $var=5400$

$T_1=\frac{L}{4},T_2=1.125L$, $var=7350$

$T_1=\frac{L}{5},T_2=1.2L$, $var=9600$

$T_1=\frac{L}{6},T_2=1.25L$,$var=11266$
