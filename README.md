# Projected WGAN-GP training

Pytorch implementation of WGAN-GP with a projection operator.

## Acknowledgements
* Modification of [link](https://github.com/jalola/improved-wgan-pytorch/)
* [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training)
* [caogang/wgan-gp](https://github.com/caogang/wgan-gp)
* [LayerNorm](https://github.com/pytorch/pytorch/issues/1959)

## Prerequisites
* Python >= 3.6
* [Pytorch v1.0.0](https://github.com/pytorch/pytorch)
* Numpy
* SciPy
* tensorboardX ([installation here](https://github.com/lanpa/tensorboard-pytorch)). It is very convenient to see costs and results during training with TensorboardX for Pytorch
* TensorFlow for tensorboardX
* Use requirements.txt to install all requirements with pip or conda

## Model

* `gan_train.py`: This model is mainly based on `GoodGenerator` and `GoodDiscriminator` of `gan_64x64.py` model from [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training). We modify this model for **polycrystalline generation** by adding a statistical projection loss and softmax activation to the generator. 

# Additional notes.
Results such as costs, generated images (every 200 iters) for tensorboard will be written to `./runs` folder.

To display the results to tensorboard, run: `tensorboard --logdir runs`


