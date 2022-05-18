import torch
import torch.nn as nn

from .twins  import Twins
from .module import Module
from .linear import Linear

def Sinkhorn_Knopp(Q, a=1, b=1, temp=.1, n_it=4):
    """
    Compute the transport map from a to b under the Gibbs kernel Q. 
    """
    # scaling factors
    Nb, Nq = Q.shape
    u = torch.ones([Nb], device=Q.device) 
    v = torch.ones([Nq], device=Q.device)
    # Sinkhorn-Knopp 
    for k in range(n_it):
        u = a / (Q @ v)
        v = b / (u.T @ Q)
    return u[:,None] * Q * v
    

class Swav (Twins):

    def __init__(self, model, energy, temp=.5, n_SK=4, buffer=1):
        super().__init__(model)
        self.temp = temp
        if isinstance(energy, tuple):
            temp = energy[2] if len(energy) > 2 else .1
            energy = Linear(*energy[:2])
        self.energy = energy
        self.buffer = torch.zeros([buffer, 1])
        self.n_SK  = n_SK
    
    def loss(self, p, z_swap):
        """
        Swapped code assignment loss.
        """
        # log-predictions 
        H = - torch.log(p)
        # swapped SK target codes 
        Z = self.update_buffer(z)
        Q = self.dense_codes(Z)
        q = Q[-z.shape[0]:].view(z.shape)
        # swapped cross-entropy loss
        return (q[:,0] * H[:,1]).sum()\
             + (q[:,1] * H[:,0]).sum()
    
    def loss_on(self, x):
        p = self(x)
        with torch.no_grad():
            z = super().forward(x)
            z_swap = torch.stack([z[:,1], z[:,0]], dim=1)
        return self.loss(p, z_swap)

    def forward (self, x):
        z = super().forward(x)
        ns = z.shape
        H = self.energy(z.view([-1, ns[-1]]))
        p = torch.softmin(H / self.temp, dim=-1)
        return H.view(ns)
    
    @torch.no_grad()
    def dense_codes (self, z, n_it=None):
        """
        Compute the transport of inputs to code densities.

        The efficient Sinkhorn-Knopp algorithm is used to minimize
        energy under the code equipartition constraint.

        The feature tensor is concatenated with the attribute 
        `swav.buffer` to facilitate code equipartition.
        """
        ns = z.shape
        n_it = n_it if n_it else self.n_SK
        # Gibbs densities
        H = self.energy(z.view([ns[0], -1]))
        H -= H.min()
        Q = torch.exp(- H / self.temp)
        Q /= Q.sum()
        # equipartition constraints
        Nb, Nq = H.shape
        Q = Sinkhorn_Knopp(Q, a=1/Nb, b=1/Nq, temp=self.temp, n_it=n_it)
        return Nb * Q.view([*ns[:-1], Nq])
    
    @torch.no_grad()
    def init_buffer (self, zs):
        """ Populate buffer with last dataset batches. """
        Nb = self.buffer.shape[0]
        self.buffer = torch.cat([zi for zi in zs[-Nb:]])
        return self.buffer
    
    @torch.no_grad()
    def update_buffer(self, z):
        """ Roll buffer with new batch. """
        self.buffer = torch.cat([self.buffer[z.shape[0]:], z])
        return self.buffer
