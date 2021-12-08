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
        xO  = (x if x.dim() == 3
                 else x.view([n_b, 1, -1]))
        xs  = [x0]
        for i in range(self.depth):
            conv = getattr(self, f'conv{i}')
            pool = getattr(self, f'pool{i}')
            xs += [pool(relu(conv(xs[-1])))]
        y = xs[-1]
        return (y if y.shape[-1] > 1
                  else y.reshape([n_b, -1]))

    """
        super().__init__()
        self.conv1 = Conv1d(1, 16, 16)     # 128 x 16
        self.pool1 = nn.MaxPool1d(4)       # 32  x 16 
        self.conv2 = Conv1d(16, 16, 8)     # 32  x 16
        self.pool2 = nn.MaxPool1d(4)       # 8   x 16
        self.conv3 = Conv1d(16, 8, 8)      # 8   x 8
        self.pool3 = nn.MaxPool1d(4)       # 4   x 8 
        self.relu  = nn.ReLU()

    def forward(self, x):
        n_b, n = x.shape
        x0 = x.view([n_b, 1, n])
        x1 = self.pool1(self.relu(self.conv1(x0)))
        x2 = self.pool2(self.relu(self.conv2(x1)))
        x3 = self.pool3(self.relu(self.conv3(x2)))
        return x3.reshape([n_b, -1])
    """
