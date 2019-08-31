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

def build_parser():
    parser = ap.ArgumentParser()
    parser.add_argument('--model', '-m', help='Path to generator', required=True)
    parser.add_argument('--outdir', '-o', help='Output directory', default='./')
    parser.add_argument('--rmin', help='Minimum p1', type=float, default=0.35)
    parser.add_argument('--rmax', help='Maximium p1', type=float, default=0.85)
    parser.add_argument('--niters', help='No. of samples to generate', type=int, default=100)
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
    p1_r = []
    p1_gl = []
    diffs = []
    for i in range(int(args.niters)):
        print('Generating sample no. {}'.format(i))
        noise = torch.randn((1, 128))
        p1 = torch.tensor(np.float32(np.random.random_sample(1))*(args.rmax - args.rmin) + args.rmin).unsqueeze(0)
        #p1 = torch.tensor([0.436]).unsqueeze(0)
        #print(p1.size())
        g_img = gen_model(noise.to(device), p1.to(device))
        g_img = g_img.detach().cpu().numpy()
        #g_img = (g_img + 1.0)/2.0
        #print(g_img.max(), g_img.min())
        #g_img[g_img>0.5] = 1.0
        #g_img[g_img<=0.5] = 0.0
        #print(p1.detach().cpu().numpy(), g_img.mean())
        p1_n = p1.detach().cpu().numpy()[0,0]
        p1_g =  g_img.mean()
        p1_gl.append(p1_g)
        p1_r.append(p1_n)
        diffs.append(p1_n-p1_g)
        #plt.imshow(g_img.reshape((64,64)), cmap='gray')
        imsave(os.path.join(args.outdir, '{}_{}_{}.png'.format(i, int(p1_n*100), int(p1_g*100))), g_img.reshape((64,64)))  
    
    with open(os.path.join(args.outdir, 'stats.pkl'),'wb') as f:
        dict_obj = {'p1':p1_r, 'g_p1':p1_gl, 'diffs':diffs}
        pkl.dump(dict_obj, f)    
    #plt.figure()
    #plt.scatter(x=p1_r, y=diffs)
    #plt.show()

if __name__ == '__main__':
    main()