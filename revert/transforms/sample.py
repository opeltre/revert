import torch 
import math
from .diff import diff

def resample(N):
    def runResample(t):
        M = t.shape[0]
        x = torch.linspace(0, 1, N)
        out = torch.zeros((N,))
        dt = diff(t)
        cub = [cubic(t[i], t[i + 1], dt[i], dt[i + 1]) \
               for i in range(M - 1)]
        for i, xi in enumerate(x):
            y = xi * (M - 1)
            j = min(math.floor(y), M - 2)
            dy = y - j
            out[i] = sum(cub[j][k] * dy ** k for k in range(4))
        return out
    return runResample

def cubic(x0, x1, v0, v1):
    a = 6 * (x1 - x0 - (v0 + v1) / 2.)
    a2 = (a + v1 - v0) / 2. 
    a3 = - a / 3  
    return torch.tensor([x0, v0, a2, a3])

def repeat(n): 
    return lambda t: torch.cat([t] * n)
