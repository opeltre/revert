import torch
import sig
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def phase3(x, t=None, ax=None, **kwargs):
    if not isinstance(t, (torch.Tensor, tuple, list)):
        t = torch.linspace(0, 1, x.shape[0])
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot(t[:-1], x[:-1], torch.diff(x), **kwargs)

def peaks(x, y, color="#fa6", sign=1, yrange=None):
    if y == None:
        y = x
        x = torch.linspace(0, 1, y.shape[0])
    y0, y1 = yrange if yrange else (min(y), max(y))    
    pks = sig.peaks(sign * y)
    plt.scatter([x[i] for i in pks], [y[i] for i in pks], color=color)
    plt.vlines([x[i] for i in pks], y0, y1, color=color, linewidth=0.5)

def jet3(x, ax=None, **kwargs):
    x = np.array(x)
    dx = np.diff(x)
    d2x = np.diff(dx)
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[:-2], dx[:-1], d2x, **kwargs)

def ax3(fig=None): 
    if not fig:
        fig = plt.figure()
    return fig.add_subplot(111, projection='3d')

def show():
    plt.show()

