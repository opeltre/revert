import torch

class Transform : 

    def __init__(self, f):
        self.callable = f

    def __call__(self, arg):
        return self.callable(arg)

    def __matmul__(self, other):
        return Transform(lambda x: self(other(x)))

    def pair(self, arg):
        return torch.stack([arg, self(arg)])

def vshift (amp):
    return Transform(
        lambda x: x + amp * torch.randn(x.shape[0])[:,None])

def noise (amp):
    return Transform(
        lambda x: x + amp * torch.randn(x.shape))

def scale (amp):
    return Transform(
        lambda x: x * (1 + amp * (torch.rand(x.shape[0]) - .5))[:,None])

def Pair (transform):
    return lambda x: torch.stack([x, transform(x)])

