import torch

from revert.models import Module

class View(Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        '''
            Reshaping the input
        '''
        out = input.view(self.shape)
        return out