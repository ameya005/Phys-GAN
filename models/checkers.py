"""
Checker functions
"""

import numpy as np
import torch
import cv2

PI = 3.1415
DIM = 64.0
SCALE = 255.0
FIXED_CIRCLE = False


class CentroidFunction(torch.nn.Module):
    def __init__(self, bs, ch, sx, sy):
        super(CentroidFunction, self).__init__()
        self.x_lin = torch.nn.Parameter(torch.linspace(0, sx, sx).expand(bs, ch, sx, sy)).requires_grad_(False).cuda()
        self.y_lin = torch.nn.Parameter(torch.linspace(0, sy, sy).expand(bs, ch, sy, sx).transpose(2, 3)
                                        ).requires_grad_(False).cuda()

    def forward(self, img_batch):
        img_batch = img_batch[:, 0:-1, ...]     # Dropping the very last channel.
        m00_t = img_batch.sum(dim=(2, 3))
        m01_t = torch.mul(img_batch, self.x_lin)
        m10_t = torch.mul(img_batch, self.y_lin)
        cx_t = torch.sum(m01_t, dim=(2, 3)) / (m00_t + 0.01)
        cy_t = torch.sum(m10_t, dim=(2, 3)) / (m00_t + 0.01)
        return cx_t, cy_t

def p1_fn(x, torch=True):
    #print(x.size())
    if torch:
        if FIXED_CIRCLE:
            return x.mean(dim=(1,2,3)).unsqueeze(1)
        else:
            #return x.mean(dim=(2,3))
            return x[:, 0:-1, ...].mean(dim=(2,3))      # Dropping the very last channel.
    else:
        return x.mean(axis=(1,2,3))

def p2_fn(x, torch=True):
    pass
