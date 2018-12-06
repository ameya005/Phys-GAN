# Phys-GANs

GANs built ot generate examples based on constraints based on physics laws.

WGAN Implementation based on the implementation by Ha Junsoo / [@kuc2477](https://github.com/kuc2477) / MIT License

## Installation
```
$ git clone https://github.com/kuc2477/pytorch-wgan-gp && cd pytorch-wgan-gp
$ pip install -r requirements.txt
```

## CLI

#### Train
```
$ # To download LSUN dataset (optional)
$ ./lsun.py --category=bedroom          

$ # To Run a Visdom server and start training on LSUN dataset.
$ python -m visdom.server
$ ./main.py --train --dataset=lsun [--resume]
```

#### Test
```
$ # checkout "./samples" directory
$ ./main.py --test --dataset=lsun
```


## References
- [Improved Training of Wasserstein GANs, arxiv:1704.00028](https://arxiv.org/abs/1704.00028)
- [caogang/wgan-gp](https://github.com/caogang/wgan-gp)


