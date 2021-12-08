import torch
import torch.nn as nn

from torch.nn.functional import relu

class Conv1d(nn.Conv1d):
    """ Periodic Convolution. """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, padding='same', padding_mode='circular', **kwargs)


class ConvNet(nn.Module):
    
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.depth  = len(layers)
        for i, ls in enumerate(zip(layers[:-1], layers[1:])):
            l0, l1 = ls
            size0, c0, w0 = l0
            size1, c1, w1 = l1
            conv = Conv1d(c0, c1, w0)
            pool = nn.MaxPool1d(size0 // size1)
            setattr(self, f'conv{i}', conv)
            setattr(self, f'pool{i}', pool)

    def forward(self, x):
        n_b = x.shape[0]
        x0  = (x if x.dim() == 3
                 else x.view([n_b, 1, -1]))
        xs  = [x0]
        for i in range(self.depth - 1):
            conv = getattr(self, f'conv{i}')
            pool = getattr(self, f'pool{i}')
            xs += [pool(relu(conv(xs[-1])))]
        y = xs[-1]
        return (y if y.shape[-1] > 1
                  else y.reshape([n_b, -1]))
