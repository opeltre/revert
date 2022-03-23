import torch

def mirror(j, bounds):
    """
    Mirror boundary conditions.
    """
    a, b = bounds
    above = (j > b).long()
    below = (j < a).long()
    return j - 2 * above * (j - b) + 2 * below * (a - j)

def bound(x, bounds=[-10, 60]):
    """
    Truncate signal within bounds.
    """
    bounds = torch.tensor(bounds)
    return torch.min(torch.max(bounds[0], x), bounds[1])
