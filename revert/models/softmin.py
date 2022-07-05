import torch
import torch.nn as nn

from .module import Module

class SoftMin (Module):
    
    def __init__(self, dim=-1, temp=None):
        super().__init__()
        if isinstance(temp, type(None)):
            self.temp = 1
        elif temp == True:
            temp = torch.tensor(1.)
            self.register_parameter('temp', nn.Parameter(temp))
        else:
            self.temp = temp

    def forward (self, x):
        return nn.functional.softmin(x / self.temp, dim=-1)

    def loss (self, py, y):
        """
        Wasserstein loss with respect to Dirac measures
        """
        Nc = y.shape[-1]
        Npts = py.shape[-1]
        Dy = torch.arange(Npts).repeat(Nc).view([Nc, Npts])
        length = (Dy - y[:,None]).abs()
        dist   = torch.min(length, Npts - length)
        return (py * dist).sum()
