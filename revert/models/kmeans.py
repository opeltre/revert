import torch
from torch.distributions.categorical import Categorical as prob

import torch 
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from .module import Module

class SimpleEuler (Optimizer):

    def __init__(self, params, lr=1.):
        super().__init__([{"params": params}], {'lr': 1.})
        self.lr = 1.

    def step(self):
        for p in self.param_groups[0]['params']:
            with torch.no_grad():
                p -= self.lr * p.grad


class KMeans(Module):
    """ K-means ++ algorithm."""
    
    def __init__(self, data, dim=1):
        """ Initialize with centers or number of centers. """
        super().__init__()
        if isinstance(data, torch.Tensor):
            self.k       = data.shape[0]
            self.centers = nn.Parameter(data)
        elif isinstance(data, int):
            self.k       = data
            self.centers = nn.Parameter(torch.zeros([self.k, dim]))
        else:
            raise TypeError("data must be int k or (k,dim) tensor")

    def predict (self, x):
        """ Predict cluster labels. """
        m = self.centers
        d = torch.cdist(x, m)
        ids = torch.sort(d, -1).indices[:,0]
        del d, m
        torch.cuda.empty_cache()
        return ids.to(x.device)

    def init (self, x):
        """ K-means ++ initialization on a dataset x."""
        i0 = torch.randint(x.shape[0], [1], device=x.device)
        m = x[i0]
        while m.shape[0] < self.k:
            d = torch.cdist(x, m)
            D = torch.sort(d, -1).values
            p = prob(probs=D[:,0] ** 2 / x.shape[-1])
            i = p.sample()
            m = torch.cat([m, x[i:i+1]])
        self.centers = nn.Parameter(m)
        del d, D, p, i, m
        torch.cuda.empty_cache()
        return self

    def loss (self, c, x):
        """ Sum of squared distances to cluster centers. """
        m = self.centers
        N = (torch.zeros([self.k], device=m.device)
                .scatter_add_(0, c, torch.ones(c.shape, device=m.device)))
        return ((x - m[c])**2 / N[c,None]).sum() / (2 * self.k * m.shape[-1])
    
    def loss_on(self, x):
        return self.loss(self(x), x)
    
    def forward (self, x):
        return self.predict(x)

    def fit(self, xs, epochs, **kws):
        """ 
        Fit on a dataset of samples xs = [x] or [x0, x1, ...].

        For true K-Means the optimizer keyword argument should 
        be time step 1 Euler scheme i.e. 

            centers -= centers.grad()
        """
        nb = kws["n_batch"] if "n_batch" in kws else 256
        if isinstance(xs, torch.Tensor):
            dset = TensorDataset(xs)
            loader = DataLoader(dset, shuffle=True, batch_size=nb)
        if not self.centers.shape[-1] == xs.shape[-1] or\
               self.centers.norm() == 0:
            self.init(xs[:nb])
        if not "optim" in kws and not "lr" in kws:
            dim = self.centers.shape[-1]
            optim = SimpleEuler(self.parameters(), lr=float(self.k * dim))
            kws["optim"] = optim
        return super().fit(loader, epochs=epochs, **kws)

    def nearest (self, n, x, pred=None):
        """ Return indices of nearest samples from cluster centers. """
        c = self.predict(x) if isinstance(pred, type(None)) else pred
        ds = torch.cdist(self.centers, x)
        _, ids = torch.sort(ds, dim=-1)
        return ids[:,:n]

    def counts (self, x, pred=None):
        """ Return tensor of predictions and number of elements per cluster."""
        c = self.predict(x) if isinstance(pred, type(None)) else pred
        Nc = (torch.zeros([self.k], device=x.device)
                .scatter_add_(0, c, torch.ones(c.shape, device=x.device)))
        return Nc

    def vars (self, x, pred=None):
        """ Clusterwise variances. """
        out = torch.zeros([self.k], device=x.device)
        m = self.centers
        c = self.predict(x) if isinstance(pred, type(None)) else pred
        N = self.counts(x)
        return ((x - m[c]) ** 2 / N[c][:,None]).sum([-1])

    def stdevs (self, x, pred=None):
        """ Clusterwise standard deviations. """
        var = self.vars(x, pred)
        return torch.sqrt(var)
    
    @torch.no_grad()
    def sort(self, x):
        Nc = self.counts(x)
        _ , idx = Nc.sort(descending=True)
        self.centers = nn.Parameter(self.centers[idx])
        self.free(Nc, idx)
        return self
