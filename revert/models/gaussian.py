import torch
import torch.nn as nn

from .module import Module
from .sinkhorn_knopp import SinkhornKnopp
from .linear import Affine


class Mixture:

    def __get__(self, obj, objtype=None):
        return obj._mix
    
    def __set__(self, obj, value):
        obj._mix = value
        obj.mix_partition = value.cumsum(0)

class Parameter:

    def __set_name__(self, obj, name):
        self.__name__ = name
        self._name = f'_{name}'

    def __get__(self, obj, objtype=None):
        val = getattr(obj, self._name)
        return val
    
    def __set__(self, obj, value):
        pass
        
class GaussianMixture (Module):

    mix = Mixture()

    def __init__(self, d, k, data=None):
        """ Gaussian k-Mixture in dimension d. """
        super().__init__()
        self.dim = d
        self.means = 3 * torch.randn(k, d)
        devsqrt = torch.eye(d) + .3 * torch.randn(k, d, d)
        self.devs = torch.einsum('...ij, ...kj -> ...ik', devsqrt, devsqrt)
        self.mix   = torch.softmax(torch.randn(k), -1)
    
    def joint(self, z):
        """
        Same as `forward` except mode indices are returned.
        """
        #--- split seed --- 
        d = z.shape[-1] - 1
        u, x = z.split([1, d], -1)
        u, x = u.contiguous().squeeze(-1), x.contiguous()
        #--- sample modes and conditional gaussian densities. 
        index = torch.bucketize(u, self.mix_partition, right=True)
        return index, self.conditional(index, x)
    
    def conditional(self, i, x): 
        """ 
        Conditional pushforward of a gaussian seed x ~ Nd(0, 1) given mode i. 

        Applies the mode-dependent affine transforms: 

            y = means[i] + devs[i] @ x 

        So that each y[i] is distributed along a multivariate gaussian centered
        at means[i] and of deviation matrix devs. 
        """
        y = self.means[i]
        dy = torch.einsum('...ij,...j -> ...i', self.devs[i], x)
        return y + dy

    def forward(self, z):
        """
        Pushforward of normalised seed z = (u, x) ~ U([0, 1]) x Nd(0, 1).

        The uniform variable u ~ U([0, 1]) is partitioned to K = {0, ..., k-1}.
        The gaussian variable is then conditionally pushed to each mode, given
        the mixture index in K.
        """
        i, y = self.joint(z)
        return y

    def loss_on(self, z_gen, x_true, temp=1, n_it=10):
        """ Wasserstein loss estimated with Sinkhorn Knopp. """
        c_gen, x_gen = self.joint(z_gen)
        cdist = torch.cdist(x_gen, x_true)
        transport = SinkhornKnopp(temp, n_it)
        Pi = transport(cdist)
        return (Pi * cdist).sum()

    def joint_sample(self, N):
        z_gen = self.seed(N)
        return self.joint(z_gen)

    def sample(self, N):
        i, y = self.joint_sample(N)
        return y
    
    def seed(self, N):
        z_gen = torch.cat([torch.rand(N, 1), torch.randn(N, self.dim)], -1)
        return z_gen