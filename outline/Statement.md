1、lllustration of VAE-CME neural network structure and training protocol

![fig1](Fig1.png)

2、Showing VAE-CME accurately predicts the distributions at times unseen in the training dataset. (three models, birth death, bursty and telegraph)

![fig1](Fig2.png)

3、Showing VAE-CME accurately predicts the distributions for unobservable species. (Oscillatory circuits, less samples and less training time)

![fig3](Fig3.png)

4、Showing VAE-CME accurately predicts distributions for reaction network with different delay mechanism. (training and predicting protocol of different delay mechanism)

![fig3](Fig4.png)


5、Showing VAE-CME accurately predicts distributions for reaction network with different kinetic parameters and topology
 <!-- prediction performance of different topology with different delay mechanism -->
在 $\tau$ 不变的情况下，用多组不同 $\rho$ 的birth death数据训练，得到的网络结构预测telegraph的双峰情况(可以)，查看是否能预测negative binomial的情况

在多组不同参数，$\tau$ 均值不变的情况下，利用Attribute对 $\tau$ 的方差进行调控，并查看是否能对telegraph的概率分布进行预测






