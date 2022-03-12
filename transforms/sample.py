import torch 
import math
from .diff import diff

def resample(N):
    """ Cubic resampling. """

    # Cubic interpolation between x0, x1 with derivatives dx0, dx1
    cubic = torch.tensor([[ 1, 0, 0, 0],
                          [ 0, 0, 1, 0],
                          [-3, 3,-2,-1],
                          [ 2,-2, 1, 1]]).float()

    def runResample(x):
        M  = x.shape[-1]
        dx = diff(x)
        # Cubic coefficients
        if not x.dim() == 1: 
            n_b, x, dx = x.shape[0], x.T, dx.T
        x_dx = torch.stack([x[:-1], x[1:], dx[:-1], dx[1:]])
        pol  = (cubic @ x_dx if x.dim() == 1 else
               (cubic @ x_dx.view([4, -1])).view([4, M - 1, n_b]))
        # Target domain 
        i = torch.arange(N)
        u = torch.linspace(0, M - 1, N)
        # Source indices
        j = torch.min(torch.floor(u), torch.tensor(M - 2)).long()
        # Offset powers
        du_pows = torch.stack([(u - j) ** k for k in range(4)])
        # Interpolation
        du_pows = du_pows if x.dim() == 1 else du_pows.unsqueeze(2)
        y = (pol[:,j] * du_pows).sum([0])
        return y if x.dim() == 1 else y.T
            
            

    return runResample

def repeat(n): 
    return lambda t: torch.cat([t] * n)
