import torch
import torch.nn as nn

from torch.nn.functional import relu
from torch import sigmoid, tanh

from .module import Module

class Conv1d(nn.Conv1d):
    """ 1D Convolutional layer with periodic boundary conditions."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, padding='same', padding_mode='circular', **kwargs)


class ConvNet(Module):
    """ 1D Convolutional network with periodic boundary conditions."""

    @classmethod
    def load(cls, path):
        """ Load model from a state file. 

            See model.save(path)
        """
        st = torch.load(path)
        return cls(**st)
    
    def __init__(self, layers=[], activation=tanh, pool='max', 
                       dropout=0.0, state=None):
        """ Create model with given layer parameters. 

            The 'layers' argument is a list of triples `[Ni, Ci, Wi]` where:
            - `Ni` is the number of time points 
            - `Ci` is the number of channels 
            - `Wi` is the width of convolution kernels
            
            A pooling layer is applied between layers i and (i + 1)
            depending on the ratio of time lengths. 
        """
        super().__init__()
        self.layers = layers
        self.depth  = len(layers)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.pool = nn.MaxPool1d if pool == 'max' else nn.AvgPool1d
        self.data = dict(layers=layers, pool=pool)
        for i, ls in enumerate(zip(layers[:-1], layers[1:])):
            l0, l1 = ls
            size0, c0, w0 = l0
            size1, c1, w1 = l1
            conv = Conv1d(c0, c1, w0)
            pool = nn.MaxPool1d(size0 // size1) 
            setattr(self, f'conv{i}', conv)
            setattr(self, f'pool{i}', pool)
        if state:
            self.load_state_dict(state)
    
    def forward(self, x):
        """ Apply model to a batch of input signals. 

            Input can be of shape [B, C, N] or [B, N] when C=1.
            Output will be of shape [B, C', N'] or [B, N'] when C'=1.
        """
        n_b = x.shape[0]
        x0  = (x if x.dim() == 3
                 else x.view([n_b, 1, -1]))
        xs  = [x0]
        a   = self.activation
        d   = self.dropout
        for i in range(self.depth - 1):
            conv = getattr(self, f'conv{i}')
            pool = getattr(self, f'pool{i}')
            xs += [pool(a(d(conv(xs[-1]))))]
        y = xs[-1]
        del xs
        torch.cuda.empty_cache()
        return (y if y.shape[-1] > 1
                  else y.reshape([n_b, -1]))

    def save (self, path):
        """ Save model data and state """
        d = self.data | {"state": self.state_dict()}
        torch.save(d, path)
