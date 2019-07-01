"""
Matsci data loader
"""

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset


class PolyCrystalDataset(Dataset):
    """
    Class to read the h5 dataset for the p1 values
    """
    def __init__(self, data_path, transform=None, mode='train'):
        super(PolyCrystalDataset, self).__init__()
        self.data = h5py.File(data_path, mode='r')['all_morph']
        self.train = mode
        if self.train == 'train':
            self.data = self.data[...,:28000]
        elif self.train == 'valid':
            self.data = self.data[...,28000:33000]
        elif self.train == 'test':
            self.data = self.data[...,33000:]

        self.transform = transform

    def __getitem__(self, index):
        x = torch.FloatTensor(self.data[..., index]).unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        #print(x.min(), x.max())
        #p1 = torch.FloatTensor(x.mean())
        return x
    
    def __len__(self):
        return self.data.shape[0]