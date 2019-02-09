"""
Checker functions
"""

import numpy as np
import torch
from matplotlib import pyplot as plt

def get_radial_indices(X):
    x, y = np.indices([X.size()[2], X.size()[3]])
    bs, ch, dimX, dimY = X.size()
    center = (int(dimX/2), int(dimY/2))
    rad = np.sqrt((x - center[0])**2 + (y-center[1])**2)
    ind = np.argsort(rad.flat)
    rad_sort = rad.flat[ind]
    rad_round = rad_sort.astype(int)
    indices =ind.reshape(1, -1)
    return indices, rad_round, rad_sort

def p1_fn(x, torch=True):
    #print(x.size())
    if torch:
        return x.mean(dim=(1,2,3)).unsqueeze(1)
    else:
        return x.mean(axis=(1,2,3))

def target_corr(x):
    rad_indices, rad_round, rad_sort = get_radial_indices(x)
    bs, ch, dimX, dimY = x.size()
    #print(dimX, dimY)
    X_fft = torch.rfft(x.permute([0,2,3,1])[:,:,:,0], signal_ndim=2, onesided=False)
    #return X_fft, 0
    X_fft_conj = torch.stack([X_fft[:,:,:,0], -1.0*X_fft[:,:,:,1]], dim=3)
    X_fft_corr = torch.mul(X_fft, X_fft_conj)
    #print((X_fft_corr[:,:,:,0] - X_fft_corr[:,:,:,1]).shape)
    #X_fft_corr = torch.stack([X_fft_corr[:,:,:,0] - X_fft_corr[:,:,:,1], torch.zeros((dimX, dimY), dtype=torch.float64)]) 
    X_fft_corr[:,:,:,0] = X_fft_corr[:,:,:,0] - X_fft_corr[:,:,:,1]
    X_fft_corr[:,:,:,1] = X_fft_corr[:,:,:,1] - X_fft_corr[:,:,:,1]
    X_corr = torch.irfft(X_fft_corr, signal_ndim=2, onesided=False)
    X_corr = torch.abs(X_corr)/(dimX * dimY)
    X_corr = torch.roll(X_corr, shifts=(int(dimX/2),int(dimY/2)), dims=(1,2))
    X_flat = X_corr.view(-1, ch*dimX*dimY)
    X_flat_sorted = X_flat[:, rad_indices].squeeze(1)
    delta_r = rad_round[1:] - rad_round[:-1]
    non_zero_deltar = np.where(delta_r > 0.0)[0]
    nind = non_zero_deltar[1:] - non_zero_deltar[:-1]
    yvals = X_flat_sorted.cumsum(dim=1)
    radial_bin = yvals[:,non_zero_deltar[1:]] - yvals[:,non_zero_deltar[:-1]]
    radial_var = radial_bin / torch.tensor(nind, dtype=torch.double)
    radial_dis = rad_round[non_zero_deltar]/(dimX*dimY)
    return radial_var, radial_dis[:-1]


def p2_fn(x, torch=True):
    if torch:
        p2_curve, p2_dist = target_corr(x)
        return p2_curve
    else:
        pass