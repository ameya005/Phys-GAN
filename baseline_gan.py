import os, sys
sys.path.append(os.getcwd())

import time
import functools
import argparse

import numpy as np
#import sklearn.datasets

import libs as lib
import libs.plot
from tensorboardX import SummaryWriter

import pdb
#import gpustat

from models.wgan import *
from models.checkers import *

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer
from matscidata import MatSciDataset
from polycrystaldata import PolyCrystalDataset

import torch.nn.init as init

# lsun lmdb data set can be download via https://github.com/fyu/lsun
# 64x64 ImageNet at http://image-net.org/small/download.php
DATA_DIR = './datasets/poly_crystals/all_polycrystalline_data.mat' # Replace your image data path here
VAL_DIR = './datasets/poly_crystals/all_polycrystalline_data.mat' 

IMAGE_DATA_SET = 'polycrystal' 
# change this to something else, e.g. 'imagenets' or 'raw' if your data is just a folder of raw images. 
# Example: 
# IMAGE_DATA_SET = 'raw'
# If you use lmdb, you'll need to write the loader by yourself. Please check load_data function

TRAINING_CLASS = ['bedroom_train'] # IGNORE this if you are NOT training on lsun, or if you want to train on other classes of lsun, then change it accordingly
VAL_CLASS = ['bedroom_val'] # IGNORE this if you are NOT training on lsun, or if you want to train on other classes of lsun, then change it accordingly

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

RESTORE_MODE = False # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0 # starting iteration 
OUTPUT_PATH = './model_outputs_polycrystal_distmap/' # output path where result (.e.g drawing images, cost, chart) will be stored
# MODE = 'wgan-gp'
DIM = 128 # Model dimensionality
CRITIC_ITERS = 5 # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 16# Batch size. Must be a multiple of N_GPUS
END_ITER = 100000 # How many iterations to train for
#END_ITER = 1
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 128*128*1 # Number of pixels in each image
PJ_ITERS = 5
INV_PARAM = 'p2' 
# def showMemoryUsage(device=1):
#     gpu_stats = gpustat.GPUStatCollection.new_query()
#     item = gpu_stats.jsonify()["gpus"][device]
#     print('Used/total: ' + "{}/{}".format(item["memory.used"], item["memory.total"]))

def proj_loss(fake_data, real_data):
    """
    Fake data requires to be pushed from tanh range to [0, 1]
    """
    if INV_PARAM == 'p1':
        return torch.abs(p1_fn(fake_data) - p1_fn(real_data))
    elif INV_PARAM == 'p2':
        return torch.norm(p2_fn(fake_data) - p2_fn(real_data))

def weights_init(m):
    if isinstance(m, MyConvo2d): 
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def load_data(path_to_folder, train):
    data_transform = transforms.Compose([
                 #transforms.Scale(64),
                 #transforms.CenterCrop(64),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                ])
    #if IMAGE_DATA_SET == 'lsun':
    #    dataset =  datasets.LSUN(path_to_folder, classes=classes, transform=data_transform)
    if IMAGE_DATA_SET == 'matsci':
        dataset = MatSciDataset(path_to_folder)
    elif IMAGE_DATA_SET == 'polycrystal':
        dataset = PolyCrystalDataset(path_to_folder, mode=train)
    else:
        dataset = datasets.ImageFolder(root=path_to_folder,transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    return dataset_loader

def training_data_loader():
    return load_data(DATA_DIR, train='train') 

def val_data_loader():
    return load_data(VAL_DIR, train='valid') 

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    #print('numel, ', real_data.numel())
    #print('real size', real_data.size() )
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    
    alpha = alpha.view(BATCH_SIZE, 1, DIM, DIM)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(BATCH_SIZE, 1, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_image(netG, noise=None, lv=None):
    if noise is None:
        noise = gen_rand_noise()
    #if lv is None:
    #    lv = torch.randn(BATCH_SIZE, 1)
    #    lv = lv.to(device)
    with torch.no_grad():
        noisev = noise
    #    lv_v = lv 
    samples = netG(noisev)
    samples = samples.view(BATCH_SIZE, 1, 128, 128)
    #samples = samples * 0.5 + 0.5
    return samples

def gen_rand_noise():
    noise = torch.randn(BATCH_SIZE, 128)
    noise = noise.to(device)

    return noise

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print(device)
fixed_noise = gen_rand_noise() 

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

if RESTORE_MODE:
    aG = torch.load(OUTPUT_PATH + "generator.pt")
    aD = torch.load(OUTPUT_PATH + "discriminator.pt")
else:
    #if INV_PARAM == 'p1':
    aG = GoodGenerator(128,128*128*1, ctrl_dim=0)
    #else:
    #    aG = GoodGenerator(64,64*64*1, ctrl_dim=44)
    aD = GoodDiscriminator(64)
    
    aG.apply(weights_init)
    aD.apply(weights_init)

LR = 1e-4
optimizer_g = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=LR, betas=(0,0.9))
#optimizer_pj = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0, 0.9))
one = torch.FloatTensor([1])
mone = one * -1
aG = aG.to(device)
aD = aD.to(device)
one = one.to(device)
mone = mone.to(device)

writer = SummaryWriter()
#Reference: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def train():
    dataloader = training_data_loader() 
    dataiter = iter(dataloader)
    for iteration in range(START_ITER, END_ITER):
        start_time = time.time()
        print("Iter: " + str(iteration))
        start = timer()
        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        try:
            real_data = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            real_data = dataiter.next()
      #  if INV_PARAM == 'p1':
      #     real_p1 = p1_fn(real_data)
      #  elif INV_PARAM == 'p2':
      #      real_p1 = p2_fn(real_data.to(device))
      #  real_p1 = real_p1.to(device)
        real_p1 = None
        for i in range(GENER_ITERS):
            print("Generator iters: " + str(i))
            aG.zero_grad()
            noise = gen_rand_noise()
            noise.requires_grad_(True)
            fake_data = aG(noise, real_p1)
            gen_cost = aD(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost.backward(mone)
            gen_cost = -gen_cost
        
        optimizer_g.step()
        end = timer()
        print(f'---train G elapsed time: {end - start}')
        print(fake_data.min(), real_data.min())
        #Projection steps
       # pj_cost = None
       # for i in range(PJ_ITERS):
       #     print('Projection iters: {}'.format(i))
       #     aG.zero_grad()
       #     noise = gen_rand_noise()
       #     noise.requires_grad=True
       #     fake_data = aG(noise, real_p1)
       #     pj_cost = proj_loss(fake_data.view(-1,1,DIM, DIM), real_data.to(device))
       #     pj_cost = pj_cost.mean()
       #     pj_cost.backward()
       #     optimizer_pj.step()


        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CRITIC_ITERS):
            print("Critic iter: " + str(i))
            
            start = timer()
            aD.zero_grad()

            # gen fake data and load real data
            noise = gen_rand_noise()
            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()
            #batch = batch[0] #batch[1] contains labels
            real_data = batch.to(device) #TODO: modify load_data for each loading
            #real_p1.to(device)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
                #if INV_PARAM == 'p1':
                #    real_p1 = p1_fn(real_data)
                #else:
                #    real_p1 = p2_fn(real_data)
                #real_p1_v = real_p1
            end = timer(); print(f'---gen G elapsed time: {end-start}')
            start = timer()
            fake_data = aG(noisev, real_p1).detach()
            end = timer(); print(f'---load real imgs elapsed time: {end-start}')
            start = timer()

            # train with real data
            disc_real = aD(real_data)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = aD(fake_data)
            disc_fake = disc_fake.mean()
            #print('fake', fake_data.size())
            #showMemoryUsage(0)
            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)
            #showMemoryUsage(0)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake  - disc_real
            
            optimizer_d.step()
            #------------------VISUALIZATION----------
            if i == CRITIC_ITERS-1:
                writer.add_scalar('data/disc_cost', disc_cost, iteration)
                #writer.add_scalar('data/disc_fake', disc_fake, iteration)
                #writer.add_scalar('data/disc_real', disc_real, iteration)
                writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)
#                writer.add_scalar('data/p1_cost', pj_cost.cpu().detach(), iteration)
                #writer.add_scalar('data/d_conv_weight_mean', [i for i in aD.children()][0].conv.weight.data.clone().mean(), iteration)
                #writer.add_scalar('data/d_linear_weight_mean', [i for i in aD.children()][-1].weight.data.clone().mean(), iteration)
                #writer.add_scalar('data/fake_data_mean', fake_data.mean())
                #writer.add_scalar('data/real_data_mean', real_data.mean())
                #if iteration %200==99:
                #    paramsD = aD.named_parameters()
                #    for name, pD in paramsD:
                #        writer.add_histogram("D." + name, pD.clone().data.cpu().numpy(), iteration)
                if iteration %10==0:
                    body_model = [i for i in aD.children()][0]
                    layer1 = body_model.conv
                    xyz = layer1.weight.data.clone()
                    tensor = xyz.cpu()
                    tensors = torchvision.utils.make_grid(tensor, nrow=8,padding=1)
                    writer.add_image('D/conv1', tensors, iteration)

            end = timer(); print(f'---train D elapsed time: {end-start}')
        #---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)

        lib.plot.plot(OUTPUT_PATH + 'time', time.time() - start_time)
        lib.plot.plot(OUTPUT_PATH + 'train_disc_cost', disc_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'train_gen_cost', gen_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'wasserstein_distance', w_dist.cpu().data.numpy())
        if iteration % 10 == 0:
            fake_2 = fake_data.view(BATCH_SIZE, 1, DIM, DIM).cpu().detach().clone()
            #fake_2 = (fake_2 + 1.0)/2.0
            fake_2 = torchvision.utils.make_grid(fake_2, nrow=8, padding=2)
            writer.add_image('G/images', fake_2, iteration)
        if iteration % 10 == 0:
            val_loader = val_data_loader() 
            #p2_vals = []
            dev_disc_costs = []
            for _, images in enumerate(val_loader):
                imgs = torch.Tensor(images[0])
                # print(imgs.size())
                imgs = imgs.to(device)
                with torch.no_grad():
                    imgs_v = imgs
                # Sample random p2's for analysis
                rn = np.random.rand()
                #if rn > 0.1 and len(p2_vals) < 64:
                #    p2_vals.append(p2_fn(imgs.unsqueeze(0)))
                D = aD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(OUTPUT_PATH + 'dev_disc_cost.png', np.mean(dev_disc_costs))
            lib.plot.flush()
            #p2_vals = torch.stack(p2_vals, dim=0).squeeze(1).to(device)
            #if p2_vals.size()[0] != BATCH_SIZE:
            #    continue
            gen_images = generate_image(aG, fixed_noise)
            torchvision.utils.save_image(gen_images, OUTPUT_PATH + 'samples_{}.png'.format(iteration), nrow=8, padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, iteration)
    #----------------------Save model----------------------
            torch.save(aG, OUTPUT_PATH + "generator.pt")
            torch.save(aD, OUTPUT_PATH + "discriminator.pt")
        lib.plot.tick()

train()


