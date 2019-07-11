"""
Generate new images
"""

import os
import numpy as np
import torch
import torchvision

import argparse as ap
from matplotlib import pyplot as plt
import pickle as pkl
import pandas as pd
from scipy.misc import imsave
import h5py
from models.checkers import *
import numpy as np

def build_parser():
    parser = ap.ArgumentParser()
    parser.add_argument('--model', '-m', help='Path to generator', required=True)
    parser.add_argument('--outdir', '-o', help='Output directory', default='./')
    parser.add_argument('--niters', help='No. of samples to generate', type=int, default=100)
    parser.add_argument('--p2_ds', help='Path of the dataset of p2 curves to sample from', default='./p2_curves.h5')
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    gen_model = torch.load(args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_model.to(device)
    gen_model.eval()
    p2_r = []
    p2_gl = []
    diffs = []
    p2_ds = h5py.File(args.p2_ds,'r')['p2s']

    ps_len = len(p2_ds)
    for i in range(int(args.niters)):
        print('Generating sample no. {}'.format(i))
        noise = torch.randn((1, 128))
        #p1 = torch.tensor(np.float32(np.random.random_sample(1))*(args.rmax - args.rmin) + args.rmin).unsqueeze(0)
        #p1 = torch.tensor([0.436]).unsqueeze(0)
        #print(p1.size())
        
        p2_idx = np.random.choice(ps_len-1)
        p2 = torch.FloatTensor(p2_ds[p2_idx])
        # print(gen_model)
        g_img = gen_model(noise.to(device), p2.to(device))
        g_img2 = g_img.detach().cpu().numpy()
        g_img = g_img.reshape((1,64,64)).unsqueeze(0)
        p2_n = p2.detach().cpu().numpy()
        p2_g =  p2_fn(g_img.cuda()).cpu().detach().numpy()
        p2_gl.append(p2_g)
        p2_r.append(p2_n)
        diffs.append(np.abs(p2_n-p2_g))
        imsave(os.path.join(args.outdir, '{}_{}_{}.png'.format(i, int(p2_n[0,0]*100), int(p2_g[0,0]*100))), g_img2.reshape((64,64)))  
        np.save(os.path.join(args.outdir, '{}_{}_{}_output.npy'.format(i, int(p2_n[0,0]*100), int(p2_g[0,0]*100))), g_img2.reshape(64,64))
    
    with open(os.path.join(args.outdir, 'stats.pkl'),'wb') as f:
        dict_obj = {'p1':p2_r, 'g_p1':p2_gl, 'diffs':diffs}
        pkl.dump(dict_obj, f)    
    #plt.figure()
    #plt.scatter(x=p1_r, y=diffs)
    #plt.show()
    #p2_ds.close()

if __name__ == '__main__':
    main()