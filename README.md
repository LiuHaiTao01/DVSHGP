Large-scale Heteroscedastic Regression via Gaussian Process
====

This is the implementation of the scalable heteroscedastic GP (HGP) developed in "*Haitao Liu,  Yew-Soon Ong, and Jianfei Cai, [Large-scale Heteroscedastic Regression via Gaussian Process](https://arxiv.org/abs/1811.01179).*" Please see the paper for further details.

We here focus on the heteroscedastic Gaussian process regression $y = f + \mathcal{N}(0, \exp(g))$ which integrates the latent function and the noise together in a unified non-parametric Bayesian framework. Though showing flexible and powerful performance, HGP suffers from the cubic time complexity, which strictly limits its application to big data. 

To improve the scalability of HGP, we first develop a variational sparse inference algorithm, named VSHGP, to handle large-scale datasets. This is performed by introducing $m$ latent variables $\mathbf{f}_m$ for $\mathbf{f}$, and $u$ latent variables $\mathbf{g}_u$ for $\mathbf{g}$. Furthermore, to enhance the model capability of capturing quick-varying features, we follow the Bayesian committee machine (BCM) formalism to distribute the learning over $M$ local VSHGP experts $\{\mathcal{M}_i\}_{i=1}^M$ with many inducing points, and aggregate their predictive distributions. At the same time, the distributed mode scales DVSHGP up to arbitrary data size!

![A toy example to illustrate DVSHGP](https://github.com/LiuHaiTao01/DVSHGP/tree/master/figs/toy.png)

This figure shows a toy example of distributed VSHGP (DVSHGP). Here, we partition the whole 500 training data into five subsets, yielding five VSHGP experts (marked with different colors) with their own inducing points for the latent function $f$ (top circles) and the noise latent function $g​$ (bottom squares). The results turn out that 

* through five distributed local experts, DVSHGP can efficiently employ up to 100 inducing points for modeling, while at the same time
*  DVSHGP successfully describes both the underlying function $f$ and the heteroscedastic noise variance (captured by $g$). 

To run the example file, execute:

```
Demo_DVSHGP_toy.m
```


