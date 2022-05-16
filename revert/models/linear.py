import torch
import torch.nn as nn

from .module import Module

class Affine(Module): 

    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.module = nn.Linear(d_in, d_out, bias)
    
    def forward(self, x):
        return self.module(x)


class Linear(Affine):
    
    def __init__(self, d_in, d_out):
        super().__init__(d_in, d_out, False)
