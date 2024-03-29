import torch
import torch.nn as nn

from torch.nn.functional import relu
from torch import sigmoid, tanh

from .module import Module
from .linear import Saxpy

class Conv1d(nn.Conv1d):
    """ 1D Convolutional layer with periodic boundary conditions."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, padding='same', padding_mode='circular', **kwargs)


class ConvNet(Module):
    """ 1D Convolutional network with periodic boundary conditions."""

    def __init__(self, layers=[], activation=tanh, pool='max', dropout=0.0, norm="Saxpy"):
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
        super().__init__()
        self.layers = layers
        self.depth  = len(layers[0])
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.pool = nn.MaxPool1d if pool == 'max' else nn.AvgPool1d
        self.data = dict(layers=layers, pool=pool)

        if len(layers) == 4:
            c, n, w, s = layers
        else:
            c, n, w = layers
            s = [1] * len(w)

        i, acts = 0, activation
        for n0, c0, w0, s0, n1, c1 in zip(n[:-1], c[:-1], w, s, n[1:], c[1:]):

            conv = Conv1d(c0, c1, w0, s0)
            setattr(self, f'conv{i}', conv)

            act = acts[i] if isinstance(acts, list) else acts
            setattr(self, f'act{i}', act)

            if norm == 'BN':
                bn = nn.BatchNorm1d(c1)
                setattr(self, f'norm{i}', bn)
            elif norm == 'Saxpy':
                saxpy = Saxpy([c1, n1])
                setattr(self, f'norm{i}', saxpy)

            pool = (nn.MaxPool1d(n0 // n1) if n0 // n1 >= 1 else
                    nn.Upsample(n1, mode='linear', align_corners=False))
            setattr(self, f'pool{i}', pool)

            i += 1

    def forward(self, x):
        """ Apply model to a batch of input signals.

            Input can be of shape [B, C, N] or [B, N] when C=1.
            Output will be of shape [B, C', N'] or [B, N'] when C'=1.
        """
        n_b = x.shape[0]
        x0  = (x if x.dim() == 3
                 else x.view([n_b, 1, -1]))
        xs  = [x0]
        d   = self.dropout
        for i in range(self.depth - 1):
            conv = getattr(self, f'conv{i}')
            a   = getattr(self, f'act{i}')
            pool = getattr(self, f'pool{i}')
            if f'norm{i}' in dir(self):
                norm = getattr(self, f'norm{i}')
                xs.append(norm(pool(a(d(conv(xs[-1]))))))
            else:
                xs.append(pool(a(d(conv(xs[-1])))))
        y = xs[-1]
        self.free(xs)
        return y

    def loss(self, y, y_tgt):
        return ((y - y_tgt)**2).mean()

    def __repr__(self):
        return f'ConvNet({self.layers})'
