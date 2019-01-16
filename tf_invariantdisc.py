#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 23:37:47 2018

@author: rahulsn
"""
import matplotlib.gridspec as gridspec
import os
from scipy import signal
from scipy import misc
import os
import matplotlib.image as mpimg
import random
from skimage.transform import resize
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt

def radial_profile(image, center, img_size):
    x, y = np.indices(image.shape)
    rad = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    ind = np.argsort(rad.flat)
    rad_sorted = rad.flat[ind]
    image_sorted = image.flat[ind]
    rad_round = rad_sorted.astype(int)
    deltar = rad_round[1:] - rad_round[:-1]
    nonzero_deltar = np.where(deltar > 0.0)[0]
    nind = nonzero_deltar[1:] - nonzero_deltar[:-1]
    yvalues = np.cumsum(image_sorted, dtype = np.float64)
    yvalues = yvalues[nonzero_deltar[1:]] - yvalues[nonzero_deltar[:-1]]
    radial_var = yvalues/nind
    radial_dis = rad_round[nonzero_deltar]/(min(image.shape))
    return radial_var, radial_dis[:-1]

def radial_profile_tf(imageRun, center, img_size):
    x, y = np.indices([img_size, img_size])
    rad = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    ind = np.argsort(rad.flat)
    rad_sorted = rad.flat[ind]
    rad_round = rad_sorted.astype(int)
    indices = ind.reshape(ind.shape[0], 1)
    imageRun_flat = tf.reshape(imageRun, [-1])
    imageRun_sorted = tf.gather_nd(imageRun_flat, indices) 
    rad_tf = tf.convert_to_tensor(rad_round)
    deltar = rad_round[1:] - rad_round[:-1]
    nonzero_deltar = np.where(deltar > 0.0)[0]
    nind = nonzero_deltar[1:] - nonzero_deltar[:-1]
    yvalues = tf.cumsum(imageRun_sorted)
    indices_for = nonzero_deltar[1:].reshape(nonzero_deltar[1:].shape[0], 1)
    indices_back =  nonzero_deltar[:-1].reshape(nonzero_deltar[:-1].shape[0], 1)
    yvalues_for = tf.gather_nd(yvalues, indices_for) 
    yvalues_back = tf.gather_nd(yvalues,indices_back) 
    radial_bin = tf.subtract(yvalues_for, yvalues_back)
    radial_var = tf.divide(radial_bin, nind)
    radial_dis = rad_round[nonzero_deltar]/img_size
    return radial_var, radial_dis[:-1]

def target_corr(images, img_size):
    radvars = list()
    for index, image in enumerate(images):
        img  = resize(image, (64, 64))
        dimX = img.shape[0]
        dimY = img.shape[1]
        fftimage = np.fft.fft2(img)
        final_image = np.fft.ifft2(fftimage*np.conj(fftimage))
        final_image_abs = np.abs(final_image)/(dimX*dimY)
        finImg = np.copy(final_image_abs)
        centrdImg = np.zeros(finImg.shape)
        for i in range(finImg.shape[0]):
            for j in range(finImg.shape[1]):
                temx = i + int(finImg.shape[0]/2)
                temy = j + int(finImg.shape[1]/2)
                if temx >= finImg.shape[0]:
                    temx = temx - finImg.shape[0]
                if temy >= finImg.shape[1]:
                    temy = temy - finImg.shape[1]
                centrdImg[i, j] = finImg[temx, temy]
        center = [int(dimX/2), int(dimY/2)]
        radvar, raddis = radial_profile(centrdImg, center, img_size)
        radvars.append(radvar)
    targets = np.array(radvars)
    return targets

def image_profile(image, img_size):
    image = tf.cast(image, dtype = tf.complex64)
    image = tf.reshape(image, [img_size, img_size])
    fft = tf.fft2d(image)
    congfft = tf.conj(fft)
    tot = fft*congfft
    ifft = tf.ifft2d(tot)
    autocorr = tf.abs(ifft)/(img_size*img_size)
    shape_at = tf.shape(autocorr)
    centrdImg = np.zeros([img_size, img_size])
    dm_hf = int(img_size/2)
    topleft = tf.slice(autocorr, [0, 0], [dm_hf, dm_hf])
    topright = tf.slice(autocorr, [0, dm_hf], [dm_hf, dm_hf])
    bottomleft = tf.slice(autocorr, [dm_hf, 0], [dm_hf, dm_hf])
    bottomright = tf.slice(autocorr, [dm_hf, dm_hf], [dm_hf, dm_hf])
    bottomhalf = tf.concat([topright, topleft], 1)
    tophalf = tf.concat([bottomright, bottomleft], 1)
    centrdImg_tf = tf.concat([tophalf, bottomhalf], 0)
    center = [int(img_size/2), int(img_size/2)]
    image_prof, image_rad = radial_profile_tf(centrdImg_tf, center, img_size)
    return image_prof
   
def corr_loss(image, targets, img_size):
    p2_prof = image_profile(image, img_size)
# Real images selection
    target_mean = np.mean(targets, axis = 0)
    p2_loss = tf.reduce_sum(tf.square(target_mean - p2_prof)) 
    return p2_loss

def batch_images(filepath, batch_size, img_size):
    epoch_count = [1]
    imagespath = filepath
    files = os.listdir(imagespath)
    tot_imgs = len(files)
    assert(tot_imgs >= batch_size)
    targets = np.zeros((tot_imgs, 64, 64)) 
    images = np.zeros((batch_size, 64, 64), dtype='int32')
    rand_files = range(tot_imgs)
    random_state = np.random.RandomState(epoch_count[0])
    random_state.shuffle(rand_files)
    epoch_count[0] += 1
    for n, file in enumerate(rand_files):
        file = imagespath+'/'+files[file]
        data = np.loadtxt(file, skiprows = 1)
        img = data[:128*128].reshape([128, 128])
        img_copy = resize(img, (64, 64))
        images[n % batch_size] = img_copy.reshape(64, 64)
        if n > 0 and n % batch_size == 0:
            return images


#Following function provides the invariance loss for p2 correlation. 

def invariance_loss(x, filepath, batch_size, img_size):
    img_size = img_size
    target_images = batch_images(filepath, batch_size, img_size)
    #target_corr is used to calculate the p2 correlation of target images (calculating the desired value of p2 correlation from calibration image). 
    #It doesn't use tf functions as there is no need of taking gradient of this function.
    target_p2 = target_corr(target_images, img_size)
    D_h1 = tf.reshape(x, [batch_size, img_size, img_size, 1])
    D_h1 = (D_h1 + 1.0)/2.0
    p2_loss = 0.0
    #in following line, the corr_loss function is used to calculate the l2 norm between the target_p2 and the p2 correlation of training images. 
    #In the function 'corr_loss', another function 'image_profile' is used - which is written using tf functions as it requires for taking a gradient. 
    p2_loss = tf.map_fn(lambda x: corr_loss(x, target_p2, img_size), D_h1)
    p2_loss = tf.reduce_sum(p2_loss)
    total_loss = p2_loss
    return total_loss
