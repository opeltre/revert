from twins import model, data

import torch
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt

st = torch.load("st/model")
model.load_state_dict(st)

def plot_weights (i): 

    layer = getattr(model, f'conv{i}')
    w = layer.weight.detach()
    b = layer.bias.detach()

    for wi, bi in zip(w, b):
        print(bi)
        plt.plot(bi + wi.T)

plt.show()
