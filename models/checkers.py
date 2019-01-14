"""
Checker functions
"""

import numpy
import torch



def p1_fn(x, torch=True):
    #print(x.size())
    if torch:
        return x.mean(dim=(1,2,3)).unsqueeze(1)
    else:
        return x.mean(axis=(1,2,3))

def p2_fn(x, torch=True):
    pass