import torch
import torch.nn as nn

from .module import Module

class Saxpy(Module):

    def __init__(self, shape, dim=-1, bias=True, stdev=.05):
        super().__init__()
        self.shape = shape
        w = 1 + stdev * torch.randn(shape)
        b = stdev * torch.randn(shape)
        self.dim = dim
        self.register_parameter('weight', nn.Parameter(w))
        self.register_parameter('bias', nn.Parameter(b))

    @torch.no_grad()
    def set(self, weight, bias):
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        if self.dim == -1 or self.dim == x.dim():
            return self.weight * x + self.bias
        if self.dim < 0:
            dlast = abs(self.dim) - 1
        else: 
            dlast = x.dim() - self.dim - len(self.shape)
        slc = [slice(None), *([None] * dlast)]
        return self.weight[slc] * x + self.bias[slc]


class Affine(Module):

    def __init__(self, d_in, d_out, dim=-1, stdev=.05, bias=True):
        super().__init__()
        self.module = nn.Linear(d_in, d_out, bias)
        self.d_in  = d_in
        self.d_out = d_out
        self.dim = dim
        self.weight = self.module.weight
        self.bias   = self.module.bias
        with torch.no_grad():
            weight = self.weight * stdev / self.weight.std()
            if self.bias is not None:
                bias = self.bias * stdev / self.bias.std()
        self.set(weight, bias)
      

    @torch.no_grad()
    def set(self, weight, bias=None):
        self.weight = nn.Parameter(weight)
        if isinstance(bias, nn.Parameter):
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

    def __init__(self, d_in, d_out, dim=-1, stdev=.05):
        super().__init__(d_in, d_out, dim, stdev, False)

    def __repr__(self):
        return f'Linear({self.d_in}, {self.d_out}, dim={self.dim})'
