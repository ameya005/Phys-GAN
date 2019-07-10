"""
Creating and plotting p2 curve as a function of p1 and corr-distance
"""

import numpy as np
from matplotlib import pyplot as plt
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--p1', type=float, help='P1 value', required=True)
parser.add_argument('--p2', type=float, required=True, help='p2 value')

args = parser.parse_args()
# volume fraction
p1_val = args.p1   # 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
# domain size
p2_val = args.p2    # 0.1, 0.2, 0.3, 0.4, 0.5

locations = np.arange(0, 1, 1. / 128)
normalized_p2_curve = lambda locs, p2: np.exp(-locs / p2)
p2_actual = lambda p1, p2_curve: p2_curve * (p1 - p1**2) + p1**2

plt.plot(locations, normalized_p2_curve(locations, p2_val))
for p1i in [0.1, 0.2, 0.3, 0.4, 0.5]:
    plt.plot(locations, p2_actual(p1i, normalized_p2_curve(locations, p2_val)))
plt.title('Same correlation length with different volume fractions')
plt.show()

plt.plot(locations, normalized_p2_curve(locations, p2_val))
for p2i in [0.1, 0.2, 0.3, 0.4, 0.5]:
    plt.plot(locations, p2_actual(p1_val, normalized_p2_curve(locations, p2i)))
plt.title('Same volume fraction with different domain sizes (p2)')
plt.show()

print('here!')