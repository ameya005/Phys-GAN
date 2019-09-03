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
from fixedcircledata import CircleDataset
import torch.nn.init as init

# lsun lmdb data set can be download via https://github.com/fyu/lsun
# 64x64 ImageNet at http://image-net.org/small/download.php
DATA_DIR = './datasets/circle/train_toyCircle_3Ch_128.h5' # Replace your image data path here
VAL_DIR = './datasets/circle/valid_toyCircle_3Ch_128.h5'

IMAGE_DATA_SET = 'circle'
# change this to something else, e.g. 'imagenets' or 'raw' if your data is just a folder of raw images. 
# Example: 
# IMAGE_DATA_SET = 'raw'
# If you use lmdb, you'll need to write the loader by yourself. Please check load_data function

torch.cuda.set_device(0)        # 0: TITANX, 1: GTX1080

TRAINING_CLASS = ['bedroom_train'] # IGNORE this if you are NOT training on lsun, or if you want to train on other classes of lsun, then change it accordingly
VAL_CLASS = ['bedroom_val'] # IGNORE this if you are NOT training on lsun, or if you want to train on other classes of lsun, then change it accordingly

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

FIXED_CIRCLE = False
NUM_CIRCLE = 2
CATEGORY = NUM_CIRCLE + 1
CONDITIONAL = True


RESTORE_MODE = False # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0 # starting iteration
# OUTPUT_PATH = './output/toyExample_radius/' # output path where result (.e.g drawing images, cost, chart) will be stored
OUTPUT_PATH = './toyExample_rad_loc_acgan/'
# MODE = 'wgan-gp'
DIM = 128 # Model dimensionality
CRITIC_ITERS = 5 # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 4
# Batch size. Must be a multiple of N_GPUS
END_ITER = 100000 # How many iterations to train for
# END_ITER = 7800
# END_ITER = 12500
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = DIM*DIM*CATEGORY # Number of pixels in each image
PJ_ITERS = 5
C = 1/DIM           # Normalizing Factor for the centroid
# def showMemoryUsage(device=1):
#     gpu_stats = gpustat.GPUStatCollection.new_query()
#     item = gpu_stats.jsonify()["gpus"][device]
#     print('Used/total: ' + "{}/{}".format(item["memory.used"], item["memory.total"]))

#def proj_loss(fake_data, real_data):
#     """
#     Fake data requires to be pushed from tanh range to [0, 1]
#     """
#    #TODO: Needs to be fixed.
#     return torch.abs(p1_fn(fake_data) - p1_fn(real_data))

centroid_fn = CentroidFunction(BATCH_SIZE, NUM_CIRCLE, DIM, DIM)  # BATCH SIZE, CH, WIDTH, HEIGHT

def proj_loss(fake_data, real_data):
    """
    Fake data requires to be pushed from tanh range to [0, 1]
    """

    if FIXED_CIRCLE:
        return torch.abs(p1_fn(fake_data) - p1_fn(real_data))
    else:
        x_fake, y_fake = centroid_fn(fake_data)
        x_real, y_real = centroid_fn(real_data)
        centerError = torch.norm(C*x_fake - C*x_real) + torch.norm(C*y_fake - C*y_real)
        # radiusError = torch.abs(p1_fn(fake_data) - p1_fn(real_data))
        radiusError = torch.norm((p1_fn(fake_data) - p1_fn(real_data)))
        return centerError + radiusError
        # return radiusError

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

def load_data(path_to_folder, classes):
    data_transform = transforms.Compose([
                 transforms.Resize(64),
                 transforms.CenterCrop(64),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                ])
    if IMAGE_DATA_SET == 'lsun':
        dataset =  datasets.LSUN(path_to_folder, classes=classes, transform=data_transform)
    elif IMAGE_DATA_SET == 'matsci':
        dataset = MatSciDataset(path_to_folder)
    elif IMAGE_DATA_SET == 'circle':
        dataset = CircleDataset(path_to_folder)
    else:
        dataset = datasets.ImageFolder(root=path_to_folder,transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    return dataset_loader

def training_data_loader():
    return load_data(DATA_DIR, TRAINING_CLASS) 

def val_data_loader():
    return load_data(VAL_DIR, VAL_CLASS) 

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    #print('numel, ', real_data.numel())
    #print('real size', real_data.size())
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    
    alpha = alpha.view(BATCH_SIZE, CATEGORY, DIM, DIM)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(BATCH_SIZE, CATEGORY, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    
    x_int, y_int = centroid_fn(interpolates)
    x_real, y_real = x_int*C, y_int*C
    int_p1 = torch.cat((x_int, y_int, p1_fn(int_data.to(device))), dim=1)
    # real_p1 = p1_fn(real_data)
    int_p1 = int_p1.to(device)
    disc_interpolates = netD(interpolates, int_p1)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_image(netG, noise=None, lv=None):
    if noise is None:
        noise = gen_rand_noise()
    if CONDITIONAL:
        if lv is None:
            if FIXED_CIRCLE:
                lv = 1 / 8 * torch.rand(BATCH_SIZE, 1)
            else:
                locationX = 30.0/DIM*torch.ones(BATCH_SIZE, NUM_CIRCLE) + 66.0/DIM*torch.rand(BATCH_SIZE, NUM_CIRCLE)
                locationY = 30.0/DIM*torch.ones(BATCH_SIZE, NUM_CIRCLE) + 66.0/DIM*torch.rand(BATCH_SIZE, NUM_CIRCLE)
                # radius = 0.13*torch.rand(BATCH_SIZE, NUM_CIRCLE) + 0.04       # For train_toyCircle_4Ch.h5
                radius = 0.12*torch.rand(BATCH_SIZE, NUM_CIRCLE) + 0.05         # For train_toyCircle_3Ch.h5
                lv = torch.cat((locationX, locationY, radius), dim=1)
                # lv = radius
            lv = lv.to(device)
    else:
        lv = None
    with torch.no_grad():
        noisev = noise
        lv_v = lv 
    samples = netG(noisev, lv_v)
    samples = torch.argmax(samples.view(BATCH_SIZE, CATEGORY, DIM, DIM), dim=1).unsqueeze(1)
    samples = (samples * 255/(CATEGORY))
    #samples = samples * 0.5 + 0.5
    return samples

def gen_rand_noise():
    noise = torch.randn(BATCH_SIZE, 128)
    noise = noise.to(device)

    return noise

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
fixed_noise = gen_rand_noise() 

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

if RESTORE_MODE:
    aG = torch.load(OUTPUT_PATH + "generator.pt")
    aD = torch.load(OUTPUT_PATH + "discriminator.pt")
else:
    if CONDITIONAL:
        if FIXED_CIRCLE:
            aG = GoodGenerator(64, DIM*DIM*1, ctrl_dim=1)
        else:
            # aG = GoodGenerator(64,64*64*1, ctrl_dim=3*NUM_CIRCLE)              # Three dimension for freeCircle
            aG = GoodGenerator(64, DIM*DIM*CATEGORY, ctrl_dim=NUM_CIRCLE+4)      # +4 for the centroid.
            aD = GoodDiscriminator(64, ctrl_dim=NUM_CIRCLE+4)
    else:
        aG = GoodGenerator(64, DIM*DIM*CATEGORY, ctrl_dim=0)
        aD = GoodDiscriminator(64)
aG.apply(weights_init)
aD.apply(weights_init)

LR = 1e-4
optimizer_g = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=LR, betas=(0,0.9))
if CONDITIONAL:
     optimizer_pj = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0, 0.9))
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

        if CONDITIONAL:
            if FIXED_CIRCLE:
                real_p1 = p1_fn(real_data)      # For fixed circle
            else:
                x_real, y_real = centroid_fn(real_data.to(device))
                x_real, y_real = x_real*C, y_real*C
                real_p1 = torch.cat((x_real, y_real, p1_fn(real_data.to(device))), dim=1)
                # real_p1 = p1_fn(real_data)
                real_p1 = real_p1.to(device)
        else:
            real_p1 = None
        for i in range(GENER_ITERS):
            print("Generator iters: " + str(i))
            aG.zero_grad()
            noise = gen_rand_noise()
            noise.requires_grad_(True)
            fake_data = aG(noise, real_p1)
            with torch.no_grad():
                fake_img = fake_data.view(-1, CATEGORY, DIM, DIM)
                x_fake, y_fake = centroid_fn(fake_img.to(device))
                x_fake, y_fake = x_fake*C, y_fake*C
                fake_p1 = torch.cat((x_fake, y_fake, p1_fn(fake_img).to(device)), dim=1)
            # real_p1 = p1_fn(real_data)
                fake_p1 = fake_p1.to(device)
            gen_cost = aD(fake_data, fake_p1)
            gen_cost = gen_cost.mean()
            gen_cost.backward(mone)
            gen_cost = -gen_cost
        
        optimizer_g.step()
        end = timer()
        print(f'---train G elapsed time: {end - start}')
        #print(fake_data.min(), real_data.min())
        if CONDITIONAL:
             #Projection steps
             pj_cost = None
             for i in range(PJ_ITERS):
                 print('Projection iters: {}'.format(i))
                 aG.zero_grad()
                 noise = gen_rand_noise()
                 noise.requires_grad=True
                 fake_data = aG(noise, real_p1)
                 pj_cost = proj_loss(fake_data.view(-1, CATEGORY, DIM, DIM), real_data.to(device))
                 pj_cost = pj_cost.mean()
                 pj_cost.backward()
                 optimizer_pj.step()


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
                if CONDITIONAL:
                    if FIXED_CIRCLE:
                        real_p1 = p1_fn(real_data)
                        real_p1 = real_p1.to(device)
                    else:
                        x_real, y_real = centroid_fn(real_data)
                        x_real, y_real = x_real*C, y_real*C
                        real_p1 = torch.cat((x_real, y_real, p1_fn(real_data)), dim=1)      # For toy example.
                        # real_p1 = p1_fn(real_data)
                        real_p1 = real_p1.to(device)
                else:
                    real_p1 = None
                #real_p1_v = real_p1
                # real_p1 = real_p1.to(device)
            end = timer(); print(f'---gen G elapsed time: {end-start}')
            start = timer()
            fake_data = aG(noisev, real_p1).detach()
            end = timer(); print(f'---load real imgs elapsed time: {end-start}')
            start = timer()

            # train with real data
            disc_real = aD(real_data, real_p1)
            disc_real = disc_real.mean()

            # train with fake data
            
            x_fake, y_fake = centroid_fn(fake_data)
            x_fake, y_fake = x_fake*C, y_fake*C
            fake_p1 = torch.cat((x_fake, y_fake, p1_fn(fake_data.view(-1, CATEGORY, DIM, DIM))), dim=1)      # For toy example.
            # real_p1 = p1_fn(real_data)
            fake_p1 = fake_p1.to(device)
            disc_fake = aD(fake_data, fake_p1)
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
                writer.add_scalar('data/disc_fake', disc_fake, iteration)
                writer.add_scalar('data/disc_real', disc_real, iteration)
                writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)
                # if CONDITIONAL:
                #     writer.add_scalar('data/p1_cost', pj_cost.cpu().detach(), iteration)
                #writer.add_scalar('data/d_conv_weight_mean', [i for i in aD.children()][0].conv.weight.data.clone().mean(), iteration)
                #writer.add_scalar('data/d_linear_weight_mean', [i for i in aD.children()][-1].weight.data.clone().mean(), iteration)
                #writer.add_scalar('data/fake_data_mean', fake_data.mean())
                #writer.add_scalar('data/real_data_mean', real_data.mean())
                #if iteration %200==99:
                #    paramsD = aD.named_parameters()
                #    for name, pD in paramsD:
                #        writer.add_histogram("D." + name, pD.clone().data.cpu().numpy(), iteration)
                # if iteration %10==0:
                #     body_model = [i for i in aD.children()][0]
                #     layer1 = body_model.conv
                #     xyz = layer1.weight.data.clone()
                #     tensor = xyz.cpu()
                #     tensors = torchvision.utils.make_grid(tensor, nrow=8,padding=1)
                #     writer.add_image('D/conv1', tensors, iteration)

            end = timer(); print(f'---train D elapsed time: {end-start}')
        #---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)

        lib.plot.plot(OUTPUT_PATH + 'time', time.time() - start_time)
        lib.plot.plot(OUTPUT_PATH + 'train_disc_cost', disc_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'train_gen_cost', gen_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'wasserstein_distance', w_dist.cpu().data.numpy())
        if iteration % 10 == 0:
            if FIXED_CIRCLE:
                fake_2 = fake_data.view(BATCH_SIZE, NUM_CIRCLE, DIM, DIM).cpu().detach().clone()
                #fake_2 = (fake_2 + 1.0)/2.0
                fake_2 = torchvision.utils.make_grid(fake_2, nrow=8, padding=2)
                writer.add_image('G/images', fake_2, iteration)
            else:
                fake_2 = torch.argmax(fake_data.view(BATCH_SIZE, CATEGORY, DIM, DIM), dim=1).unsqueeze(1)
                fake_2 = (fake_2 * 255 / (CATEGORY))
                fake_2 = fake_2.int()
                fake_2 = fake_2.cpu().detach().clone()
                fake_2 = torchvision.utils.make_grid(fake_2, nrow=8, padding=2)
                writer.add_image('G/images', fake_2, iteration)
        if iteration % 10 == 0:
            val_loader = val_data_loader() 
            dev_disc_costs = []
            for _, images in enumerate(val_loader):
                imgs = torch.Tensor(images[0])
                imgs = imgs.to(device)
                with torch.no_grad():
                    imgs_v = imgs
                # real_p1 = p1_fn(real_data)
                x_real, y_real = centroid_fn(imgs_v)
                x_real, y_real = x_real*C, y_real*C
                real_p1 = torch.cat((x_real, y_real, p1_fn(imgs_v)), dim=1)      # For toy example.
                #real_p1 = p1_fn(real_data)
                real_p1 = real_p1.to(device)
                #fake_p1 = fake_p1.to(device)
                D = aD(imgs_v, real_p1)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(OUTPUT_PATH + 'dev_disc_cost.png', np.mean(dev_disc_costs))
            lib.plot.flush()
            gen_images = generate_image(aG, fixed_noise)
            torchvision.utils.save_image(gen_images, OUTPUT_PATH + 'samples_{}.png'.format(iteration), nrow=8, padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, iteration)
    #----------------------Save model----------------------
            torch.save(aG, OUTPUT_PATH + "generator.pt")
            torch.save(aD, OUTPUT_PATH + "discriminator.pt")
        lib.plot.tick()

train()


