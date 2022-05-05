import torch

from .module import Module

class SoftMin (Module):

    def forward (self, x):
        return torch.softmin(x, dim=-1)

    def loss (self, py, y):
        """
        Wasserstein loss with respect to Dirac measures
        """
        Nc = y.shape[-1]
        Npts = py.shape[-1]
        Dy = torch.arange(Npts).repeat(Nc).view([Nc, Npts])
        return (py * (Dy - y[:, :, None]).abs()).sum()
