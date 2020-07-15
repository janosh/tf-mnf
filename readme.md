# MNF-BNN

TensorFlow 2.0 implementation of Multiplicative Normalizing Flows (MNF): a Bayesian neural network variant introduced in [[1]](#mnf-bnn) with auxiliary random variables and a factorial Gaussian posterior with means `mu_i` conditioned on scaling factors `z_i` modelled by hypernets that drive a normalizing flows.

Implements the planar [[3]](#vi-nf), RNVP [[4]](#rnvp) and MAF [[5]](#maf) flow. MAF uses the MADE autoregressive network architecture introduced in [[2]](#made)

## References

1. <a id="mnf-bnn"></a> **_Multiplicative Normalizing Flows for Variational Bayesian Neural Networks_** | Christos Louizos, Max Welling (Mar 2017) | [1703.01961](https://arxiv.org/abs/1703.01961)

2. <a id="made"></a> **_MADE: Masked Autoencoder for Distribution Estimation_** | Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle (Jun 2015) | [1502.03509](https://arxiv.org/abs/1502.03509)

3. <a id="vi-nf"></a> **_Variational Inference with Normalizing Flows_** | Danilo Rezende, Shakir Mohamed (May 2015) | [1505.05770](https://arxiv.org/abs/1505.05770)

4. <a id="rnvp"></a> **_Density estimation using Real NVP_** | Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio (May 2016) | [1605.08803](https://arxiv.org/abs/1605.08803)

5. <a id="maf"></a> **_Masked Autoregressive Flow for Density Estimation_** | George Papamakarios, Theo Pavlakou, Iain Murray (Jun 2018) | [1705.07057](https://arxiv.org/abs/1705.07057)

6. <a id="bay-hyp"></a> **_Bayesian Hypernetworks_** | David Krueger, Chin-Wei Huang, Riashat Islam, Ryan Turner, Alexandre Lacoste, Aaron Courville (Oct 2017) | [1710.04759](https://arxiv.org/abs/1710.04759)

## Environment

The environment file `env.yml` was generated with `conda env export --no-builds > env.yml`. To recreate the environment from this file run `conda env create -f env.yml`.

The environment `flownn` was originally created by running the command:

```sh
conda create -n flownn python=3.6 jupyter tqdm seaborn \
  && conda activate flownn \
  && pip install tensorflow tensorflow-probability pre-commit
```

To delete the environment run `conda env remove -n flownn`.

To update all packages and reflect changes in this file use

```sh
conda update --all \
  && pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U \
  && conda env export --no-builds > env.yml
```
