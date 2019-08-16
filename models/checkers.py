"""
Checker functions
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
# from gan_train import BATCH_SIZE

DIM = 64
CATEGORY = 6
PI = 3.1416
BATCH_SIZE = 64
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
    radial_var = radial_bin/torch.FloatTensor(nind).cuda()
    # radial_var = radial_bin/torch.DoubleTensor(nind).cuda()
    radial_dis = rad_round[non_zero_deltar]/(dimX*dimY)
    return radial_var, radial_dis[:-1]

def p2_fn(x, torch=True):
    if torch:
        p2_curve, p2_dist = target_corr(x)
        return p2_curve
    else:
        pass

def normalized_p2(p2, epsilon=1e-8):
    """

    :param p2: p2-curve
    :return: normalized p2-curve
    """
    return (p2 - p2[-1, :]) / (p2[0, :]-p2[-1, :]+epsilon)

def grain_regularize_fn(x, label):
    objective_tensor = torch.zeros(BATCH_SIZE, 44).cuda()
    # with torch.no_grad():
    for i in range(CATEGORY):
        p2_curve = p2_fn(x[:, i].unsqueeze(1).float())  # (Batch, 1, 64, 64) to (Batch, 44)
        normalized_p2_curve = normalized_p2(p2_curve)   # (Batch, 44)

        # r_star = torch.sqrt(p2_curve[0, :] * (DIM*DIM/PI/n[i]))
        r_star = torch.sqrt(p2_curve[:, 0] * (DIM*DIM/PI/(label/20).squeeze(1)))       #

        exp = torch.exp(-torch.arange(44).repeat(1, BATCH_SIZE).view(BATCH_SIZE, -1).float().cuda()/r_star.unsqueeze(1))

        difference = exp - normalized_p2_curve
        objective_tensor += torch.abs(difference)

    return objective_tensor


