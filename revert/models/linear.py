import torch
import torch.nn as nn

from .module import Module

class Affine(Module): 

    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.module = nn.Linear(d_in, d_out, bias)
        self.d_in  = d_in
        self.d_out = d_out
    
    def forward(self, x):
        if self.d_in == x.shape[-1]:
            return self.module(x)
        elif self.d_in == 1 and self.d_out == 1:
            b, w = self.module.bias, self.module.weight
            return b.flatten() + w.flatten() * x
        raise TypeError(f"Invalid input shape {x.shape} for d_in={self.d_in}")

class Linear(Affine):
    
    def __init__(self, d_in, d_out):
        super().__init__(d_in, d_out, False)
