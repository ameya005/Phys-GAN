"""
Matsci data loader
"""

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset


class MatSciDataset(Dataset):
    """
    Class to read the h5 dataset for the p1 values
    """
    def __init__(self, data_path, transform=None):
        super(MatSciDataset, self).__init__()
        self.data = h5py.File(data_path, mode='r')['morphology_64_64']
        self.transform = transform

    def __getitem__(self, index):
        x = torch.FloatTensor(self.data[index, ...])
        if self.transform is not None:
            x = self.transform(x)
        #print(x.min(), x.max())
        #p1 = torch.FloatTensor(x.mean())
        return x/255.
    
    def __len__(self):
        return self.data.shape[0]