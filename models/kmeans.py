import torch
from torch.distributions.categorical import Categorical as prob

class KMeans:
    """ Implementation of k-means ++ for pytorch."""
    
    @classmethod
    def load (cls, data):
        return cls(torch.load(data))

    def __init__(self, data):
        """ Initialize with means or number of means. """
        if isinstance(data, torch.Tensor):
            self.k       = data.shape[0]
            self.centers = data
        elif isinstance(data, int):
            self.k       = data
            self.centers = None
        else:
            raise TypeError
    
    def predict (self, x):
        """ Predict cluster labels. """
        m = self.centers
        d = torch.cdist(x, m)
        _, ids = torch.sort(d, -1)
        return ids[:,0]

    def init (self, x):
        """ K-means ++ initialization on a dataset x."""
        i0 = torch.randint(x.shape[0], [1], device=x.device)
        m = x[i0]
        while m.shape[0] < self.k:
            d = torch.cdist(x, m)
            D, ids = torch.sort(d, -1)
            p = prob(D[:,0] ** 2)
            i = p.sample()
            m = torch.cat([m, x[i:i+1]])
        self.centers = m
        return self

    def fit (self, x, eps=1e-5, n_it=1000, verbose=False, sort=True):
        """ Fit on a dataset x. """
        if isinstance(self.centers, type(None)): 
            self.init(x)
        d = x.shape[-1]
        # centroids
        m  = torch.zeros([self.k, x.shape[-1]], device=x.device)
        # number of elements per class
        Nc   = torch.zeros([self.k],  device=x.device)
        ones = torch.ones(x.shape[0], device=x.device)
        for n in range(n_it):
            # E step
            c = self.predict(x)
            # M step 
            Nc.zero_().scatter_add_(0, c, ones)
            idx = c[:,None].repeat(1, d)
            m.zero_().scatter_add_(0, idx, x)
            m /= Nc[:,None]
            # Convergence criterion
            dm = m - self.centers
            if verbose: print(dm.norm())
            if dm.norm() < eps:
                if verbose: print(f"converged at step {n}")
                break
            self.centers = m + 0
        if sort:
            Nc = self.counts(x)
            _ , ids = Nc.sort(descending=True)
            self.centers = self.centers.index_select(0, ids)
        return self
    
    def nearest (self, n, x, pred=None):
        """ Return indices of nearest samples from cluster centers. """
        c = self.predict(x) if isinstance(pred, type(None)) else pred
        ds = torch.cdist(self.centers, x)
        _, ids = torch.sort(ds, dim=-1)
        return ids[:,:n]

    def classes (self, x, pred=None):
        """ Return list of classes. """
        c = self.predict(x) if isinstance(pred, type(None)) else pred
        C = []
        for j in range(self.k):
            in_j = labels == j
            id_j = torch.nonzero(in_j)[:,0]
            x_j = x.index_select(0, id_j)
            C += [x_j]
        return C

    def loss (self, x, pred=None):
        """ Sum of squared distances to cluster centers. """
        m = self.centers
        c = self.predict(x) if isinstance(pred, type(None)) else pred
        mc = m.index_select(0, c)
        return ((x - mc)**2).sum() / x.shape[0]

    def vars (self, x, pred=None):
        """ Clusterwise variances. """
        out = torch.zeros([self.k], device=x.device)
        m = self.centers
        c = self.predict(x) if isinstance(pred, type(None)) else pred
        mc = m.index_select(0, c)
        d2 = ((x - mc) ** 2).sum([-1])
        Nc = torch.zeros([self.k]).scatter_add_(0, c, torch.ones(c.shape))
        return out.scatter_add_(0, c, d2) / Nc

    def stdevs (self, x):
        """ Clusterwise standard deviations. """
        return torch.sqrt(self.vars(x))

    def counts (self, x, pred=None):
        """ Return tensor of predictions and number of elements per cluster."""
        c = self.predict(x) if isinstance(pred, type(None)) else pred
        Nc = (torch.zeros([self.k], device=x.device)
                .scatter_add_(0, c, torch.ones(c.shape, device=x.device)))
        return Nc
    
    def cuda(self):
        """ Move means tensor to cuda. """
        self.centers = self.centers.cuda()
        return self

    def cpu(self):
        """ Move means tensor to cuda. """
        self.centers = self.centers.cpu()
        return self

    def save (self, path):
        torch.save(self.centers, path)

    def __repr__(self):
        tail = "(None)" if isinstance(self.centers, type(None)) else ""
        return f"{self.k}-Means {tail}"

    def swap (self, i, j):
        mi, mj = self.centers[i], self.centers[j]
        self.centers[i] = mj
        self.centers[j] = mi
        return self
