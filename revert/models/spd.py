import torch
import torch.nn as nn

from .module import Module
from torch.nn.utils.parametrizations import orthogonal

class MetricParameter:

    def __set_name__(self, obj, name):
        self.__name__ = name
        self._name = f'_{name}'

    def __get__(self, obj, objtype=None):
        val = getattr(obj, self._name)
        return val
    
    def __set__(self, obj, value):
        pass

class SPD(Module):
    """ Container class for SPD matrices. """

    def __init__(self, d, k=1, vals=None, vecs=None):
        super().__init__()
        self.dim = d
        self.n_matrices = k
        #--- eigenvalues ---
        if isinstance(vals, type(None)):
            vals = (.8 * torch.randn(k, d)).abs()
        #--- eigenvectors ---
        if isinstance(vecs, type(None)):
            vecs = torch.randn(k, d, d)
        self.vals = nn.Parameter(vals)
        self.vecs = nn.Parameter(vecs)
        orthogonal(self, 'vecs')

    def matrices(self):
        u = self.vecs 
        D = torch.exp(self.vals)[:,:,None] * torch.eye(self.dim)
        uD = torch.matmul(u.transpose(1, 2), D)
        S = torch.matmul(uD, u)
        return S
    
    def det(self):
        return torch.exp(self.vals.sum([-1]))
    
    def inv_matrices(self):
        u = self.vecs 
        D = torch.exp(- self.vals)[:,:,None] * torch.eye(self.dim)
        uD = torch.matmul(u.transpose(1, 2), D)
        S = torch.matmul(uD, u)
        return S
        
    def forward(self, x, index=None):
        S = self.matrices()
        S = (S if isinstance(index, type(None)) 
                    else S[index])
        return torch.matmul(S, x.unsqueeze(-1)).squeeze(-1)

    def reverse(self, x, index=None):
        S = self.inv_matrices()
        S = (S if isinstance(index, type(None)) 
                    else S[index])
        return torch.matmul(S, x.unsqueeze(-1)).squeeze(-1)
        
