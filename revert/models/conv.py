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
        if not len(layers) >= 3:
            raise RuntimeError('Layers arguments must be of length 3 or 4')
        
        self.layers     = layers
        self.depth      = len(layers[0]) - 1
        #--- Convolution parameters 
        self.channels   = layers[0]
        self.Npoints    = layers[1]
        self.widths     = layers[2]
        self.strides    = (layers[3] if len(layers) == 4 
                                     else [1] * self.depth)
        self.groups     = (groups if groups is not None 
                                  else [1] * self.depth)
        #--- Activations
        self.activations = (act if isinstance(act, (list, tuple)) 
                                else [act] * self.depth)
        #--- Pooling
        self.pool = pool
        #--- Normalization Layers (BN or Saxpy)
        self.norms      = (norm if isinstance(norm, (list, tuple)) 
                                else [norm] * self.depth)
        
        modules = []
        for i in range(self.depth):
            layer = self.layer(i)
            modules.append(layer)

        super().__init__(*modules)
    
    def layer(self, i):
        """ Composed layer number i """
        pipe = [self.conv_layer(i),
                self.activation_layer(i),
                self.pool_layer(i),
                self.norm_layer(i)]
        layer = Pipe(*[f for f in pipe if f is not None])
        return layer
        
    def conv_layer(self, i, Cin=None, Cout=None):
        """ i-th Convolutional layer. """
        C = self.channels 
        w, s, g  = self.widths[i], self.strides[i], self.groups[i] 
        Cin = C[i]      if Cin is None else Cin
        Cout = C[i+1]   if Cout is None else Cout
        return Conv1d(Cin, Cout, w, s, groups=g)
    
    def pool_layer(self, i):
        """ i-th Pooling layer. """
        N = self.Npoints
        ratio = N[i] // N[i+1]
        if self.pool == 'max' and ratio >= 1:
            return nn.MaxPool1d(ratio)
        elif self.pool == 'avg' and ratio >= 1:
            return nn.AvgPool1d(ratio)
        elif ratio < 1:
            return nn.Upsample(N[i+1], mode='linear', align_corners=False)
    
    def activation_layer(self, i):
        """ i-th Activation layer. """
        return self.activations[i]
    
    def norm_layer(self, i, Cout=None):
        """ i-th Normalization layer (BacthNorm or Saxpy). """
        norm = self.norms[i]
        if isinstance(norm, str):
            Cout = self.channels[i+1] if Cout is None else Cout
            Nout = self.Npoints[i+1]
            if norm == 'Saxpy':
                return Saxpy([Cout], -2)
            if norm == 'BN':
                return nn.BatchNorm1d(Cout)
        return norm
            
    def loss(self, y, y_tgt):
        return ((y - y_tgt)**2).mean()

    def __repr__(self):
        return f'ConvNet({self.layers})'
