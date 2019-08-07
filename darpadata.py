"""
Matsci data loader
"""

import torch
import h5py
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset


class DarpaDataset(Dataset):
    """
    Class to read the h5 dataset for the p1 values
    """

    def __init__(self, data_path, transform=None, train='train'):
        super(DarpaDataset, self).__init__()
        #self.data = h5py.File(data_path, mode='r')['dataset']
        self.data = np.load(data_path)
        self.data[self.data==0] = 1
        self.data = np.array(self.data) - 1                 # Forcing to have label 0 to 5.
        self.data = self.data.astype(int)                   # every element should be int.
        print(self.data.min(), self.data.max())
        if train == 'train':
            self.data = self.data[:800, ...]
        elif train == 'valid':
            self.data = self.data[800:1200, ...]
        else:
            self.data = self.data[1200:,...]
        self.data = torch.from_numpy(self.data)             # converting to tensor
        #self.data = self.data.transpose(1, 3)
        self.transform = transform

    def __getitem__(self, index):
        # x = torch.FloatTensor(self.data[index, ...])
        x = self.data[index, ...]
        if self.transform is not None:
            x = self.transform(x)
        x = F.one_hot(x, num_classes=6)                    # one_hot
        x = x.transpose(0,2)
        # print(x.min(), x.max())
        # p1 = torch.FloatTensor(x.mean())
        return x

    def __len__(self):
        return self.data.shape[0]

# data_path = './datasets/voronoi/orient_voronoi_noZero.h5'
# v = VoronoiDataset(data_path=data_path)
#
# print(v[0])
