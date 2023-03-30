import torch
import torch.nn as nn

from .module import Module, Slice, Prod, Branch
from .jet import Jet

class Wavelet(Module):

    def __init__(self, dim, c_in, c_out, w, s=1, p=None):
        super().__init__()
        Conv_d = getattr(nn, f'Conv{dim}d')
        self.conv = Conv_d(c_in, c_out, w, stride=s, bias=False)
        self.dim = dim
        self.c_in = c_in
        self.c_out = c_out
        self.width = w
        self.stride = s
    
    @property
    def weight(self):
        return self.conv.weight
    
    @weight.setter
    def weight(self, val):
        self.conv.weight = nn.Parameter(val)
    
    def forward(self, x):
        Ns, d = list(x.shape), self.dim
        d_in = -1 - d
        if x.shape[-d - 1] == self.c_in:
            return self.conv(x)
        elif self.c_in == 1:
            return self.conv(x.unsqueeze(d_in))
    
    def jet(self, rank):
        """ Jets of wavelet kernels. """
        return JetWavelet(self.dim, rank, self)
    
    def downsample(self, stride):
        """ Strided convolution with same weights. """
        d, c_in, c_out, w = self.dim, self.c_in, self.c_out, self.width
        W = Wavelet(d, c_in, c_out, w, s=stride)
        with torch.no_grad():
            W.weight = self.weight.detach()
        return W


class JetWavelet (Wavelet):

    def __init__(self, dim, rank, wavelet):
        #--- Jet of wavelet kernels 
        J = Jet(dim, rank)
        Jw = J(wavelet.weight)
        c_out = Jw.shape[0]
        #--- Wavelet instance
        super().__init__(dim, 1, c_out, wavelet.width, wavelet.stride)
        self.weight = Jw.unsqueeze(1)
        self.Jet = J
        self.rank = rank
        #--- Pointers to graded components  
        self.indices = J.indices
        self.sizes = torch.tensor([len(ij) for ij in J.indices])
        self.begin = [0, *self.sizes.cumsum(0)]

    def amplitudes(self, jet):
        """ 
        Amplitude of jets accross graded components. 
        """
        print(jet.shape)
        dim = -1 - self.dim
        slices = [b + torch.arange(n) for b, n in zip(self.begin, self.sizes)]
        j_split = [jet.index_select(dim, slc) for slc in slices]
        j_norm = [(jk ** 2).sum([dim]).sqrt() for jk in j_split]
        return torch.stack(j_norm, dim)
    
    def energies(self):
        pass

    def grades(self, k0, k1=None):
        begin = self.begin
        if k1 is None: 
            k1 = self.rank + 1
        if k1 < 0: 
            k1 = self.rank + 1 + k1 
        slices = [slice(begin[i], begin[i+1]) for i in range(k0, k1)]
        return (Prod(*[Slice(slc, -1 -self.dim) for slc in slices])
                @ Branch(len(slices))
                @ self)
    

class Heat (Wavelet):
    """ Gaussian Kernel. """

    def __init__(self, dim, width, idev=3):
        """ 
        Convolution by a gaussian kernel. 
        """
        self.dim = dim
        self.width = width
        d, r = dim, width
        super().__init__(dim, 1, 1, width)
        #--- Gaussian kernel 
        gauss = torch.ones([width] * dim)
        x = torch.linspace(-idev, idev, width)
        gx = torch.exp(-(x**2)/2)
        for i in range(d):
            gx_d = gx[(slice(None), *([None] * i))]
            gauss *= gx_d 
        gauss /= gauss.sum()
        self.weight = gauss
        self.freeze()