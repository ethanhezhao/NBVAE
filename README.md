# NBVAE

This is the demo code for [NBVAE](https://arxiv.org/abs/1905.00616).

# Datasets

The datasets of 20NG and ML-10M are provided.

The other text data can be downloaded from the code repo of [DPFA](https://github.com/zhegan27/dpfa_icml2015).

The other collaborative-filtering data can be downloaded from the links in the paper and preprocessed with the code of [MultiVAE](https://github.com/dawenl/vae_cf).

# Installation & Set-Up

The code is implemented with Python 3.5.2 and Tensorflow 1.10.0, and also requires Numpy, Scipy, Scikit-learn, Pandas, and Bottleneck installations.

# Run the demos

The demos of NBVAE and NBVAE_dm on text data are in ```demo_NBVAE.sh```.

The demo of NBVAE_b on binary collaborative-filtering data is in ```demo_NBVAE_b.sh```.


