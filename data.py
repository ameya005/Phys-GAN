"""
Data related functions
"""
import torch
from torchvision import transforms
from matscidata import MatSciDataset

# ==================
# Dataset Transforms
# ==================


_MAT_SCI_TRANSFORMS = [
    transforms.Normalize(mean=[0.0], std=[1.0])
]

# ========
# Datasets
# ========

TRAIN_DATASETS = {
    'matsci': lambda: MatSciDataset(
        './datasets/matsci/morph_global_64_train_255.h5',
        transform= transforms.Compose(_MAT_SCI_TRANSFORMS))
}
TEST_DATASETS = {
    'matsci': lambda: MatSciDataset(
        './datasets/matsci/morph_global_64_valid_255.h5',
        transform= transforms.Compose(_MAT_SCI_TRANSFORMS))
}

DATASET_CONFIGS = {
    'matsci': {'size': 64, 'channels':1}
}
