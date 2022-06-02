# Robust Training under Label Noise by Sparse Over-parameterization (SOP)

[![Paper](https://img.shields.io/badge/paper-arXiv%3A2007.00151-green)](https://arxiv.org/abs/2007.00151)

</div>

This repository is the official implementation of [Robust Training under Label Noise by Over-parameterization](https://arxiv.org/abs/2007.00151) (**ICML 2022**).

We propose a principled approach for robust training of over-parameterized deep networks in classification tasks where a proportion of training labels are corrupted. The main idea is yet very simple: label noise is sparse and incoherent with the network learned from clean data, so we model the noise and learn to separate it from the data. Specifically, we model the label noise via another sparse over-parameterization term, and exploit implicit algorithmic regularizations to recover and separate the underlying corruptions. Remarkably, when trained using such a simple method in practice, we demonstrate state-of-the-art test accuracy against label noise on a variety of real datasets. Furthermore, our experimental results are corroborated by theory on simplified linear models, showing that exact separation between sparse noise and low-rank data can be achieved under incoherent conditions. The work opens many interesting directions for improving over-parameterized models by using sparse over-parameterization and implicit regularization.

### Example
For 40% symmetric noise
```
python python train.py -c config_cifar100.json --percent 0.4
```


```
@article{liu2022robust,
  title={Robust Training under Label Noise by Over-parameterization},
  author={Liu, Sheng and Zhu, Zhihui and Qu, Qing and You, Chong},
  journal={arXiv preprint arXiv:2202.14026},
  year={2022}
}
```
