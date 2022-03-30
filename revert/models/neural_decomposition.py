import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.fft          import rfft, irfft

from math import pi

class ND (nn.Module):
    
    @staticmethod
    def map(params, Npts=128): 
        """ 
        Map N_b x 2 x N_modes params to a N_b x N_pts tensor by IFFT.
        """
        t = torch.linspace(0, 2*pi, Npts)
        n_b, _, n_modes = params.shape
        w = torch.arange(1, n_modes + 1).float()
        amp, phi = params[:,0,:], params[:,1,:]
        wt    = w[:,None] @ t[None,:]
        modes = torch.sin(wt[None,:,:] + phi[:,:,None])
        return (amp[:,:,None] * modes).sum(dim=[1])

    def __init__(self, n_modes):
        super().__init__()
        self.n_modes = n_modes
        #--- Parameters ---
        self.amp = Parameter(torch.zeros([n_modes]))
        self.w   = Parameter(torch.zeros([n_modes]))
        self.phi = Parameter(torch.zeros([n_modes]))

    def forward (self, t):
        wt = self.w[:, None] @ t[None, :]
        modes = torch.sin(wt + self.phi[:,None])
        return (self.amp[:, None] * modes).sum(dim = [0])

    def fit (self, x, lr=0.001, n_it=1000, br=0.001):
        for s in range(n_it):
            y = self.forward(x) 
            loss = torch.sqrt(torch.sum((y - x)**2))
            loss.backward()
            with torch.no_grad():
                for p in self.parameters(): 
                    p -= p.grad * lr
                    p -= br * torch.randn(p.shape)
                self.zero_grad()
        return self

    def init (self, x, dev=0.01): 
        Fx = rfft(x)
        _, js = torch.sort(Fx.abs(), descending=True)
        n = self.n_modes
        phi = torch.index_select(Fx, 0, js).angle()[:n]
        w   = js[:n]
        amp = dev *  torch.randn([n]).abs() * torch.sqrt(torch.var(x))
        self.phi = Parameter(phi)
        self.w   = Parameter(w.float())
        self.amp = Parameter(amp)
        return self
