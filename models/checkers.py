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

class CentroidFunction(torch.nn.Module):
    def __init__(self, bs, ch, sx, sy):
        super(CentroidFunction, self).__init__()
        self.x_lin = torch.nn.Parameter(torch.linspace(0, sx, sx).expand(bs, ch, sx, sy)).requires_grad_(False)
        self.y_lin = torch.nn.Parameter(torch.linspace(0, sy, sy).expand(bs, ch, sy, sx).transpose(2,3)).requires_grad_(False)
    
    def forward(self, img_batch):
        m00_t = img_batch.sum(dim=(2,3))
        m01_t = torch.mul(img_batch, self.x_lin)
        m10_t = torch.mul(img_batch, self.y_lin)
        cx_t = torch.sum(m01_t, dim=(2,3))/(m00_t+0.01)
        cy_t = torch.sum(m10_t, dim=(2,3))/(m00_t+0.01)
        return cx_t, cy_t