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
    
    def __init__(self, density, dim=-1, window=None, strict=False):
        """ Create filter from spectral density. """
        self.density = density
        self.size    = self.density.shape[0]
        self.window  = window
        self.strict  = strict
        self._cached = {}
    
    def cache(self, N, real=True):
        """ Cache resampled density acting on length N signals. """
        if real:
            if N % 2 != 0 and self.strict: print('irfft @ rfft != 1 for odd N')
            n = N // 2 + 1
        self._cached[N] = (resample(n)(self.density),
                           resample(N)(self.window) if self.window else 1)
        return self

    def __call__(self, arg):
        """ Apply filter on a (batched) signal. """
        N = arg.shape[-1]
        if not N in self._cached:
            # cache or print or error
            if self.strict:
                print(f"caching sparse operator for size {N}")
            self.cache(N)
        # read cache
        d, w = self._cached[N]
        if d.dim() == arg.dim(): 
            F_arg = rfft(w * arg)
            return irfft(d * F_arg)
        # batched
        F_arg = rfft(w * arg, dim=1)
        return irfft(d * F_arg, dim=1)

    def __repr__(self):
        keys = [str(k) for k in self._cached.keys()]
        skeys = "{" + ", ".join(keys) + "}"
        return f"Convolution kernel {skeys}"

def bandpass (F0, F1, Fs, N=100, h=0):
    # real frequency range [0, Fs / 2]
    i0, i1 = floor(2 * N * F0 / Fs), floor(2 * N * F1 / Fs)
    w = step(i0, i1, N)
    w = heat(h)(w) if h > 0 else w
    return Spectral(w)

def lowpass (Fcut, Fs, N=100, h=0):
    return bandpass(0, Fcut, Fs, N, h)

def highpass (Fcut, Fs, N=100, h=0):
    return bandpass(Fcut, Fs, Fs, N, h)


#---- Spatial Filters ----


class Spatial: 
    """ Convolution Kernels """

    def __init__(self, density, strict=False):
        """ Create filter from convolution kernel. """
        self.strict = strict
        self.density = density
        self.size = self.density.shape[0]
        self._cached = {}

    def __getitem__(self, N):
        return self._cached[N]

    def clear_cache(self):
        """ Clear cached matrices. """ 
        self._cached = {}
        return self

    def cache (self, N):
        """ Cache a sparse matrix acting on length N signals. """
        d = self.size 
        stride = torch.arange(d) - d // 2
        #--- Kernel indices
        i = torch.arange(N)[None,:].repeat(d, 1)
        j = i + stride[:,None]
        #--- Mirror boundaries
        right = (j >= N).long() * 2 * (N - 1 - j)
        left  = (j < 0).long()  * 2 * (- j)
        j = left + j + right
        #--- Kernel Values
        values = self.density[:,None].repeat(1, N).flatten()
        #--- Cache sparse operator
        indices = torch.stack([i, j]).view([2, -1])
        mat = torch.sparse_coo_tensor(indices, values, size=[N, N])
        self._cached[N] = mat
        return self

    def __call__(self, arg): 
        """ Apply filter to a (batched) signal. """
        N = arg.shape[-1]
        if not N in self._cached: 
            # cache or print or error
            if self.strict:
                print(f"caching sparse operator for size {N}")
            self.cache(N)
        # read cache
        mat = self._cached[N]
        if arg.dim() == self.density.dim():
            return mat @ arg
        # batched
        n_b = arg.shape[0]
        return torch.sparse.mm(mat, arg.T).T

    def __repr__(self):
        keys = [str(k) for k in self._cached.keys()]
        skeys = "{" + ", ".join(keys) + "}"
        return f"Convolution kernel {skeys}"

def heat (radius, fs=1):
    """ Heat kernel """
    diam = 2 * radius
    n = floor(2 * fs * diam + 1)
    x = torch.linspace(-diam, diam, n)
    gx = torch.exp(- x**2 / (2 * radius ** 2))
    return Spatial(gx / gx.sum())

