import icp.loader as hdf5
import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def phase3(x, t=None, **kwargs):
    if not isinstance(t, (torch.Tensor, tuple, list)):
        t = torch.linspace(0, 1, x.shape[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(t[:-1], x[:-1], torch.diff(x), **kwargs)

def show():
    plt.show()

def jet3(x, **kwargs):
    x = np.array(x)
    dx = np.diff(x)
    d2x = np.diff(dx)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[:-2], dx[:-1], d2x, **kwargs)
