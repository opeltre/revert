import torch
import torch.nn as nn

from torch.nn.functional import relu
from torch import sigmoid, tanh

from .module import Module, Pipe, Id
from .linear import Saxpy

#tanh = nn.Tanh()

class Conv1d(nn.Conv1d):
    """ 1D Convolutional layer with periodic boundary conditions."""

    def __init__(self, *args, **kwargs):
        super().__init__(
                *args, padding='same', padding_mode='circular', **kwargs)

    def forward(self, x):
        if x.dim() == 3:
            return super().forward(x)
        n_b = x.shape[0]
        return super().forward(x.view([n_b, 1, -1]))

class ConvNet(Pipe):
    """ 1D Convolutional network with periodic boundary conditions."""

    def __init__(self, layers=[], groups=None, pool='max', act=tanh, norm='Saxpy', **kws):
        """
        Create model with given layer parameters.

        The 'layers' argument is either a triplet [C, N, W] or
        quadruplet [C, N, W, S] of lists where:
        - `N[i]` is the number of time points on layer i
        - `C[i]` is the number of channels on layer i
        - `W[i]` is the width of convolution kernels on layer i (i < depth)
        - `S[i]` is the stride applied on layer i (i < depth)

        A pooling or upsampling layer is applied between layers i and (i + 1)
        depending on the ratio of time lengths.
        """
        self.layers = layers
        self.depth  = len(layers[0])
        
        if len(layers) == 4:
            c, n, w, s = layers
        elif len(layers) == 3:
            c, n, w = layers
            s = [1] * len(w)
        else: 
            raise RuntimeError('Layers arguments must be of length 3 or 4')

        if groups is None:
            groups = [1] * len(w)

        modules = []
        i = 0
        for n0, c0, w0, s0, n1, c1, g in zip(n[:-1], c[:-1], w, s, n[1:], c[1:], groups):

            conv = Conv1d(c0, c1, w0, s0, groups=g)
            act  = act[i] if isinstance(act, list) else act
            
            if norm == 'BN':
                norm = nn.BatchNorm1d(c1)
            elif norm == 'Saxpy':
                norm = Saxpy([c1, n1])
            else:
                norm = Id()
            
            pool = (nn.MaxPool1d(n0 // n1) if n0 // n1 >= 1 else
                    nn.Upsample(n1, mode='linear', align_corners=False))

            i += 1
        
            layer = Pipe(conv, act, pool, norm)
            modules.append(layer)

        super().__init__(*modules)


    def loss(self, y, y_tgt):
        return ((y - y_tgt)**2).mean()

    def __repr__(self):
        return f'ConvNet({self.layers})'
