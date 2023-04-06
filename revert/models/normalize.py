import torch

from .module import Module

class Normalize (Module):
    """ Normalize along last dimensions. """

    def __init__(self, dim=1, center=True, p=2, conv=None):
        super().__init__()
        self.dim = dim
        self.conv = None
        self.p = p
        self.center = center

    def joint(self, x):
        d  = self.dim
        y = x.flatten(start_dim=-d)
        N = y.shape[-1]
        #--- expand last dimension
        slc = (*[slice(None)] * (y.dim() - 1), None)
        if self.center:
            y0 = y.mean([-1])
            y = y - y0[slc]
        #--- Normalise energy density
        uy = y.abs() ** self.p
        Ey = uy.mean([-1])
        u = y / Ey[slc]
        #--- Remove channel dimensions and restore dim
        return ((u.view(x.shape), y0, Ey) if self.center else
                (u.view(x.shape), 0, Ey))

    def forward(self, x):
        u, mean, E = self.joint(x)
        return u