import torch

from .module import Module

class View(Module):
    """ 
    Module reshaping its input, preserving batch dimension. 
    """

    def __init__(self, shape):
        """ 
        Create module. 

        Input: 
            - shape : sizes after the batch dimension.
        """
        super().__init__()
        self.shape = shape

    def forward(self, input):
        """ Reshape the input. """
        out = input.view([input.shape[0], *self.shape])
        return out

    def __repr__(self):
        return f'View({self.shape})'
