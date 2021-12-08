import torch 
import math
from .diff import diff

def resample(N):

    def runResample(x):
        M   = x.shape[0]
        t  = torch.linspace(0, 1, N) * (M - 1)
        t0 = torch.min(t.floor(), torch.tensor(M - 2)[None])
        dt = torch.stack([(t - t0) ** k for k in range(4)])
        cub = cubic(x, diff(x)).index_select(1, t0.long())
        return (cub * dt).sum(dim=0)

    return runResample

def cubic(x, dx):
    x0, x1  = x[:-1],  x[1:]
    v0 , v1 = dx[:-1], dx[1:]
    a = 6 * (x1 - x0 - (v0 + v1) / 2.)
    a2 = (a + v1 - v0) / 2. 
    a3 = - a / 3  
    return torch.stack([x0, v0, a2, a3])

def repeat(n): 
    return lambda t: torch.cat([t] * n)
