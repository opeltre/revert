import torch
from torch.distributions.categorical import Categorical as prob

import torch 
import torch.nn as nn
from torch.optim import Optimizer

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
        return ids

    def init (self, x):
        """ K-means ++ initialization on a dataset x."""
        i0 = torch.randint(x.shape[0], [1], device=x.device)
        m = x[i0]
        while m.shape[0] < self.k:
            d = torch.cdist(x, m)
            D = torch.sort(d, -1).values
            p = prob(logits=(D[:,0] ** 2))
            i = p.sample()
            m = torch.cat([m, x[i:i+1]])
        self.centers = nn.Parameter(m)
        del d, D, p, i, m
        torch.cuda.empty_cache()
        return self

    def loss (self, c, x):
        """ Sum of squared distances to cluster centers. """
        m = self.centers
        N = torch.zeros([self.k]).scatter_add_(0, c, torch.ones(c.shape))
        return ((x - m[c])**2 / N[c,None]).sum() / 2
    
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
        if not self.centers.shape[-1] == xs[0].shape[-1]:
            self.init(xs[0])
        if not "optim" in kws and not "lr" in kws:
            optim = SimpleEuler(self.parameters(), lr=1.)
            kws["optim"] = optim
        return super().fit(xs, epochs=epochs, **kws)
