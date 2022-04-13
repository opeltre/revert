import torch
import torch.nn as nn

from .module import Module

def norm2 (t, dim=None):
    """ L2-norm on specified dimensions """
    return torch.sqrt((t ** 2).sum(dim)) 

def cross_correlation (ya, yb):
    """ Cross-correlation of N_batch x N """
    ya, yb = ya - ya.mean([0]), yb - yb.mean([0])
    yab = ya[:,:,None] @ yb[:,None,:]
    return yab.sum(dim=[0]) / (norm2(ya, [0]) * norm2(yb, [0]))


class BarlowTwins (Module):

    def __init__(self, model, diag=2):
        """ Create twins from a model. """
        super().__init__()
        self.model  = model
        self.diag   = diag 
        self.writer = False

    def forward (self, x):
        """ Apply twins to 2 x N_batch x N tensor. """
        xa, xb = x
        ya, yb = self.model(xa), self.model(xb)
        return torch.stack([ya, yb])

    def loss (self, y): 
        """ Return Barlow twin loss of N_batch x N output. """
        n_out = y.shape[-1]
        C = cross_correlation(*y) 
        I = torch.eye(n_out, device=C.device)
        w = self.diag
        loss_mask = 1 + (w - 1) * I
        return torch.sum(((C - I) * loss_mask) ** 2) / (n_out ** 2)

    def cross_corr (self, x):
        """ Cross correlation matrix of twin outputs. """
        return cross_correlation(*self(x))
