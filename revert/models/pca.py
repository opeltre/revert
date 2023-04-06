import torch
import torch.nn as nn

import torch.nn.utils.parametrize as P
from torch.nn.utils.parametrizations import orthogonal

from .module import Module

class PCA (Module):

    def __init__(self, shape, rank=None, dim=0):
        super().__init__()
        self.shape = tuple(shape)
        self.dim = dim
        self.d_in = rank 
        self.d_out = int(torch.tensor(shape).prod())
        
        #--- parameters ---
        mean = torch.randn(*shape)
        diag = torch.ones(rank) + .1 * torch.randn(rank)
        vecs = torch.randn(rank, self.d_out)
        self.mean = nn.Parameter(mean)
        self.vals = nn.Parameter(diag.sort(0, True).values)
        self.vecs = nn.Parameter(vecs)
        orthogonal(self, 'vecs')
        
    def right_inverse(self, y):
        """ Projection onto principal components. """
        y = y.transpose(0, self.dim) if self.dim else y
        y = y - self.mean
        y = y.flatten(start_dim=1)
        x = (y @ self.vecs.T) / self.vals
        return x 

    def forward(self, x):
        """ Embedding of dominant subspaces. """
        y = (self.vals * x) @ self.vecs
        y = y.view([x.shape[0], *self.shape])
        y = y + self.mean
        return y.transpose(0, self.dim) if self.dim else y
    
    def right_loss(self, y_true):
        """ Reconstruction loss. """
        y_gen = self(self.right_inverse(y_true))
        print(y_gen.var(), y_true.var())
        return (y_gen - y_true).var() / y_true.var()
    
    def init(self, y):
        """ Initialize PCA on a point cloud. """
        y = y.transpose(0, self.dim) if self.dim else y
        y = y.flatten(start_dim=1)
        y0 = y.mean([0])
        dy = y - y0
        Q = (dy.T @ dy) / y.shape[0]
        vals, vecs = torch.linalg.eigh(Q)
        vals, idx = vals.sort(0, True)
        with torch.no_grad():
            self.mean = nn.Parameter(y0.view(self.shape))
            self.vals = nn.Parameter(vals[:self.d_in])
            self.vecs = vecs.T[idx[:self.d_in]].contiguous()



        
