"""

Toy example (circle) data loader.
"""

import torch
import h5py
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset

class CircleDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(CircleDataset, self).__init__()
        self.data = h5py.File(data_path, mode='r')['dataset']
        self.transform = transform

    def __getitem__(self, index):
        x = torch.FloatTensor(self.data[index, ...])
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.data.shape[0]