import torch
import torch.nn as nn

from .module import Module

class Saxpy(Module):

    def __init__(self, shape, bias=True, stdev=.1):
        super().__init__()
        self.shape = shape
        w = 1 + stdev * torch.randn(shape)
        b = stdev * torch.randn(shape)
        self.register_parameter('weight', nn.Parameter(w))
        self.register_parameter('bias', nn.Parameter(b))

    @torch.no_grad()
    def set(self, weight, bias):
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return self.weight * x + self.bias


class Affine(Module):

    def __init__(self, d_in, d_out, dim=-1, bias=True):
        super().__init__()
        self.module = nn.Linear(d_in, d_out, bias)
        self.d_in  = d_in
        self.d_out = d_out
        self.dim = dim
        self.weight = self.module.weight
        self.bias   = self.module.bias

    @torch.no_grad()
    def set(self, weight, bias=None):
        self.weight = nn.Parameter(weight)
        if not isinstance(bias, type(None)):
            self.bias = nn.Parameter(bias)

    def forward(self, x):
        if self.d_in == 1 and self.d_out == 1:
            b, w = self.module.bias, self.module.weight
            y = (w.flatten() * x
                 + (b.flatten() if isinstance(b, torch.Tensor) else 0))
            return y
        if self.dim != -1:
            x = x.transpose(self.dim, -1)
        ns = x.shape
        if self.d_in == x.shape[-1]:
            y = self.module(x.reshape([-1, ns[-1]]))
            y = y.reshape([*ns[:-1], self.d_out])
            return y if self.dim == -1 else y.transpose(-1, self.dim)
        raise TypeError(f"Invalid shape {x.shape} for d_in={self.d_in}")

    def __repr__(self):
        return f'Affine({self.d_in}, {self.d_out}, dim={self.dim})'

class Linear(Affine):

    def __init__(self, d_in, d_out, dim=-1):
        super().__init__(d_in, d_out, dim, False)

    def __repr__(self):
        return f'Linear({self.d_in}, {self.d_out}, dim={self.dim})'
