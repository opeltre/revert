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
        """ Apply kernels to batched inputs. """
        Ns, d = list(x.shape), self.dim
        d_in = -1 - d
        #--- flatten batch dimensions 
        if x.shape[d_in] == self.c_in:
            y = self.conv(x.flatten(end_dim=d_in - 1))
            return y.view([*Ns[:d_in], *y.shape[d_in:]])
        #--- unsqueeze 1-channel inputs
        elif self.c_in == 1:
            y = self.conv(x.unsqueeze(d_in).flatten(end_dim=-d-2))
            return y.view([*Ns[:d_in+1], *y.shape[d_in:]])
    
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
        Jw = J(wavelet.weight.flatten(end_dim=-dim-1))
        c_out = Jw.shape[0]
        #--- Wavelet instance
        super().__init__(dim, 1, c_out, wavelet.width, wavelet.stride)
        if not Jw.dim() == dim + 2:
            Jw = Jw.unsqueeze(1)
        Jw = Jw.transpose(-dim -1, -dim - 2)
        self.weight = Jw.flatten(end_dim=1).unsqueeze(1)
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
   
    def slices(self, k0=0, k1=None):
        begin = self.begin
        if k1 is None: 
            k1 = self.rank + 1
        if k1 < 0: 
            k1 = self.rank + 1 + k1 
        slcs = [slice(begin[i], begin[i+1]) for i in range(k0, k1)]
        return (Prod(*[Slice(slc, -1 -self.dim) for slc in slcs])
                @ Branch(len(slcs)))
    
    def forward(self, x):
        y = super().forward(x)
        d = self.dim
        n = self.begin[-1]
        if n != self.c_out:
            Ns, Ms = y.shape[:-d-1], y.shape[-d:]
            return y.view([*Ns, -1, n, *Ms])
        else:
            return y


    def grades(self, k0=0, k1=None):    
        return self.slices(k0, k1) @ self
    

class Heat (Wavelet):
    """ Gaussian Kernel. """

    def __init__(self, dim, width, idev=3):
        """ 
        Convolution by gaussian kernels. 
        """
        self.dim = dim
        self.width = width
        d, r = dim, width
        #--- inverse deviations 
        idev  = torch.tensor(idev).view([-1]).float()
        c_out = idev.numel()
        super().__init__(dim, 1, c_out, width)
        #--- Gaussian kernels
        gauss = torch.ones(c_out, *([width] * dim))
        t = torch.linspace(-1, 1, width)
        x = idev[:,None] * t
        gx = torch.exp(-(x**2)/2)
        for i in range(d):
            slc = (slice(None), *([None] * (d-1-i)),
                   slice(None), *([None] * i))
            gx_d = gx[slc]
            gauss *= gx_d 
        Z = gauss.flatten(start_dim=-d).sum([-1])
        print(Z.shape)
        gauss /= Z[(slice(None), *([None] * d))]
        self.weight = gauss.unsqueeze(1)
        self.freeze()