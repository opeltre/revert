import torch
import math
from torch.fft import rfft, irfft

class Filter: 

    def __init__(self, density, ftype='spatial'):
        self.ftype = ftype if ftype == 'spatial' else 'spectral'
        self.density = density
        self.size = self.density.shape[0]

    def __call__(self, arg): 
        if self.ftype == 'spatial':
            diam = self.size
            r = math.floor(diam / 2)
            stack = torch.stack([
                self.density[i] * translate(r - i, arg)
                for i in range(diam)
            ])
            return stack.sum(0)

def translate(i, t):
    return torch.cat((t[:i].flip(0), t.roll(i)[i:]))\
        if i >= 0\
        else torch.cat((t.roll(i)[:i], t[i:].flip(0)))


class Heat(Filter): 

    def __init__(self, radius, fs=1): 
        diam = 2 * radius
        n = math.floor(2 * fs * diam + 1)
        x = torch.linspace(- diam, diam, n)
        gx = torch.exp(- x**2 / (2 * radius ** 2))
        super().__init__(gx / gx.sum(), 'spatial')
