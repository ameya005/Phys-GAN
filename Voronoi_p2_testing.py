import torch

from voronoidata import VoronoiDataset
from matscidata import MatSciDataset
import numpy as np
# from matplotlib import pyplot as plt
from models.checkers import p2_fn

torch.manual_seed(0)
np.random.seed(0)
# Dataset
ds = VoronoiDataset(data_path='./datasets/voronoi/valid_6000_lhs_20.h5')
# ds = MatSciDataset(data_path='./datasets/matsci/morph_global_64_valid_255.h5')
numChannel = 6

def p2_hat(p2):
    """

    :param p2: p2-curve
    :return: normalized p2-curve
    """
    epsilon = 1e-8
    return (p2[0] - p2[0][-1]) / (p2[0][0]-p2[0][-1]+epsilon)

def normalized_p2(p2, epsilon=1e-8):
    """

    :param p2: p2-curve
    :return: normalized p2-curve
    """
    return (p2 - p2[:, -1]) / (p2[:][0]-p2[:][-1]+epsilon)


noise = torch.randn(1, 128)
noise = noise.to(torch.device("cuda"))
model = torch.load("./voronoi_output/generator.pt")

image = model(noise).reshape(6, 64, 64)

test = torch.argmax(image.view(1, 6, 64, 64), dim=1).unsqueeze(1)
test = test.squeeze(0)
test = test.squeeze(0).cpu()


# for i in range(6):
#     print(len(np.where(test == i)[0]))
#     temp = p2_fn(image[i].unsqueeze(0).unsqueeze(0).float().cuda())
#     print(temp)



for i in range(6):
    print(len(np.where(ds[0][i] == 1)[0]))
    temp = p2_fn(ds[0][i].unsqueeze(0).unsqueeze(0).float().cuda())
    print(temp.shape)
    # temp = p2_hat(temp)
    temp = normalized_p2(temp)
    print(temp.shape)
    # numGrain = temp[0][1] / channel_area(ds[0][i])

# temp = p2_fn(ds[0].unsqueeze(0).cuda())
#     print(temp)


