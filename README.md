# Robust Training under Label Noise by Sparse Over-parameterization (SOP)

[![Paper](https://img.shields.io/badge/paper-arXiv%3A2007.00151-green)](https://arxiv.org/abs/2202.14026)

</div>

This repository is the official implementation of [Robust Training under Label Noise by Over-parameterization](https://proceedings.mlr.press/v162/liu22w.html) (**ICML 2022**).

We propose a principled approach for robust training of over-parameterized deep networks in classification tasks where a proportion of training labels are corrupted. The main idea is yet very simple: label noise is sparse and incoherent with the network learned from clean data, so we model the noise and learn to separate it from the data. Specifically, we model the label noise via another sparse over-parameterization term, and exploit implicit algorithmic regularizations to recover and separate the underlying corruptions. Remarkably, when trained using such a simple method in practice, we demonstrate state-of-the-art test accuracy against label noise on a variety of real datasets. Furthermore, our experimental results are corroborated by theory on simplified linear models, showing that exact separation between sparse noise and low-rank data can be achieved under incoherent conditions. The work opens many interesting directions for improving over-parameterized models by using sparse over-parameterization and implicit regularization.

### Example
Please follow Table A.1 for hyperparameters. 
For 50% symmetric noise
```
python train.py -c config_cifar100.json --lr_u 1 --lr_v 10 --percent 0.5
```
For 40% Asymetric noise （Because we modified the code for better delivery after camera ready, for asymmetric noise we use lr for u equals 0.1 rather than 1 in the paper)
```
python train.py -c config_cifar100.json --lr_u 0.1 --lr_v 100 --percent 0.4 --name CIFAR100 --asym True
```
```
@InProceedings{pmlr-v162-liu22w,
  title = 	 {Robust Training under Label Noise by Over-parameterization},
  author =       {Liu, Sheng and Zhu, Zhihui and Qu, Qing and You, Chong},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {14153--14172},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR}
}
```
