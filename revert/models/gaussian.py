import torch
import torch.nn as nn

from .module import Module
from .sinkhorn_knopp import SinkhornKnopp
from .spd import SPD
from .linear import Affine


class Mixture:

    def __get__(self, obj, objtype=None):
        return obj._mix
    
    def __set__(self, obj, value):
        obj._mix = nn.Parameter(value)
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
        self.k   = k
        self.means = nn.Parameter(10 * torch.randn(k, d))
        self.devs = SPD(d, k)
        self.mix  = torch.softmax(torch.randn(k), -1)
    
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
        dy = self.devs(x, i)
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

    def joint_likelihood(self, i, x):
        """ 
        Likelihood of pairs (i, x) of latent and observed samples.

        Given modes `i : [*Ns]` and vectors `x : [*Ns, d]`, returns
        the likelihood tensor `p(x|i) : [*Ns]`. 
        """
        Ns, N = i.shape, i.numel()
        i, x = i.flatten(), x.view([N, self.dim])
        #--- Quadratic energies ---
        dx = x - self.means[i]
        ds = self.devs.reverse(dx, i)
        Hx = (ds * ds).sum([1]) / 2
        #--- Gaussian normalisation factors ---   
        Z = self.devs.det()[i] 
        Z = Z * (torch.tensor(2 * torch.pi) ** self.dim).sqrt()
        #--- Joint likelihoods --- 
        p = self.mix[i] * torch.exp(- Hx) / Z
        return p.view([*Ns])
    
    def lift_samples(self, x_true):
        """ 
        Lift observed samples to pairs (i, x), repeating x. 

        Given batched vectors `x_true : [N, d]`, will return 
        a tuple `(i, x)` of batched modes and vectors
        `i : [N, k]` and `x : [N, k, d]` where `x` is constant 
        accross the first dimension. 
        """
        k, d, n =  self.k, self.dim, x_true.shape[0]
        x = x_true.repeat_interleave(k, dim=0).view([n, k, d])
        i = torch.arange(k).repeat(n).view([n, k])
        return i, x

    def predict(self, x_true):
        """ 
        Posterior probabilities on mode indices.
        """
        k, d, n =  self.k, self.dim, x_true.shape[0]
        i, x = self.lift_samples(x_true)
        p = self.joint_likelihood(i, x)
        p_x = p / p.sum([1])[:,None]
        return p_x
    
    def log_likelihood(self, x_true, eps=1e-6):
        """ Expected log-likelihood of an observed sample. """
        i, x = self.lift_samples(x_true)
        p = self.joint_likelihood(i, x)
        logpx = (p * torch.log(p + eps)).sum([1])
        return logpx
    
    def likelihood(self, x_true):
        """ Marginal likelihood of an observed sample. """
        i, x = self.lift_samples(x_true)
        p = self.joint_likelihood(i, x)
        return p.sum([1])

    def transport(self, x_gen, x_true, temp=1, n_it=10):
        """ Sinkhorn-Knopp optimal transport. """
        trans = SinkhornKnopp(temp, n_it)
        cdist = torch.cdist(x_gen, x_true)
        Pi = trans(cdist)
        return Pi

    def loss_on(self, z_gen, x_true, temp=1, n_it=10):
        """ Wasserstein loss estimated with Sinkhorn Knopp. """
        c_gen, x_gen = self.joint(z_gen)
        cdist = torch.cdist(x_gen, x_true)
        #--- mix-aware source distribution ---
        mix = self.mix[c_gen]
        with torch.no_grad():
            _mix = 0. + mix.data
        A = mix / (_mix * x_gen.shape[0])
        #--- optimal transport ---
        transport = SinkhornKnopp(temp, n_it)
        Pi = transport(cdist, A=A)
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