import torch
from torch.fft import rfft, irfft
from math import floor, ceil

from .sample import resample

#---- Spectral Filters ----

def step (i0, i1, N):
    """ Step function """
    return torch.cat(
        [torch.zeros([i0]), torch.ones([i1 - i0]), torch.zeros([N - i1])])

class Spectral:
    """ Spectral Filters """
    
    def __init__(self, density, dim=-1, window=None):
        self.density = density
        self.size    = self.density.shape[0]
        self.window  = window

    def __call__(self, arg):
        w = resample(arg.shape[0])(self.window) if self.window else 1
        F_arg = rfft(w * arg)
        if F_arg.shape[0] != self.size:
            print(f"resampling from {self.size} to {F_arg.shape[0]}")
            d = resample(F_arg.shape[0])(self.density)
        else:
            d = self.density
        return irfft(d * F_arg)

def bandpass (F0, F1, Fs, N=100, h=0):
    i0, i1 = floor(2 * N * F0 / Fs), floor(2 * N * F1 / Fs)
    w = step(i0, i1, N)
    w = heat(h)(w) if h > 0 else w
    return Spectral(w)

def lowpass (Fcut, Fs, N=100, h=0):
    return bandpass(0, Fcut, Fs, N, h)

def highpass (Fcut, Fs, N=100, h=0):
    return bandpass(Fcut, Fs, Fs, N, h)


#---- Spatial Filters ----


def translate (i, t):
    """ Translate by relative indices """
    return torch.cat((t[:i].flip(0), t.roll(i)[i:]))\
        if i >= 0\
        else torch.cat((t.roll(i)[:i], t[i:].flip(0)))


class Spatial: 
    """ Convolution kernels """

    def __init__(self, density):
        self.density = density
        self.size = self.density.shape[0]

    def __call__(self, arg): 
        diam = self.size
        r = floor(diam / 2)
        stack = torch.stack([
            self.density[i] * translate(r - i, arg)
            for i in range(diam)
        ])
        return stack.sum(0)

def heat (radius, fs=1):
    """ Heat kernel """
    diam = 2 * radius
    n = floor(2 * fs * diam + 1)
    x = torch.linspace(-diam, diam, n)
    gx = torch.exp(- x**2 / (2 * radius ** 2))
    return Spatial(gx / gx.sum())

