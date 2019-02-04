"""
FESolver Module

This module uses scipy modules
* Requires an image with a fixity, force.
The FESolver module is used to calculate strain energy

TODO:

1. Add the Forward and backward functions to an StrainEnergy(nn.Module) for calculating the loss
2. Minimize the total strain energy
3. Figure out how to use this across batches of data
4. Maybe move the forward and backward functions to pycuda or similar.
"""

import torch
from torch.autograd import Function as F
from torch import nn
from FESolver import ForwardTopo, BackwardTopo

from torch.autograd.gradcheck import gradcheck

class TopoCheckerFunction(F):
    """
    Topology checker. 
    Takes in a generated map and calculates the displacement and strain energy metrics
    """
    @staticmethod
    def forward(ctx, input):
        # TODO: Define input parameters
        pass
    
    @staticmethod
    def backward(ctx, grad):
        # TODO: Define gradient parameters
        pass

class StrainEnergy(nn.Module):
    """
    Calculate the total strain energy and a strain energy map.
    Minimizing the strain energy under the constraint of a specific fixity
    and force.
    """
    def __init__(self):
        super().__init__()


