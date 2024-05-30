### Introduction

基因表达研究的复杂性，发展的很快，CME已经成为了基因表达建模的重要方法

已经有很多利用CME解决基因表达建模的方法，以及存在的缺陷 1、大多数都是做的马尔可夫过程，非马尔可夫的很少。2、对时滞的研究很少。3、在多种物质的系统中，许多建模方法无法使用一种物质的数据预测另一种物质的数据。4、基因表达结构之间的变换研究很少，特别是是否有Switch的存在

变分自编码器的特点

In our study，

### Results

#### lllustration of VAE-CME neural network structure and training protocol

Introduce CME with delay and how to solve it with neural network

How does VAE work

![fig1](Fig1.png)

#### VAE-CME accurately predicts the distributions at times unseen in the training dataset. (three models, birth death, bursty and telegraph)

描述三个反应
![fig1](Fig2.png)

#### VAE-CME accurately predicts the distributions for unobservable species. (Oscillatory circuits, less samples and less training time)

描述震荡反应
![fig3](Fig3.png)

#### VAE-CME accurately predicts distributions for reaction network with different delay mechanism. (training and predicting protocol of different delay mechanism)

![fig3](Fig4.png)

#### VAE-CME accurately predicts distributions for reaction network with different kinetic parameters and topology

### Discussion

<!-- prediction performance of different topology with different delay mechanism -->

在 $\tau$ 不变的情况下，用多组不同 $\rho$ 的birth death数据训练，得到的网络结构预测telegraph的双峰情况(可以)，查看是否能预测negative binomial的情况

在多组不同参数，$\tau$ 均值不变的情况下，利用Attribute对 $\tau$ 的方差进行调控，并查看是否能对telegraph的概率分布进行预测

1、先把实验做了
2、盘一下整个文章的思路，让我自己写会怎么写
3、着重看一下introduction怎么做，寻找别的文章的弱点以及我这个文章的优势
4、仔细看PNAS上那篇文章
5、图的配色看下
